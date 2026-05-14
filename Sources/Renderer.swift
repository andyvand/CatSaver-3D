import Metal
import QuartzCore
import AppKit
import simd

/// A Metal 4 renderer for the cat screensaver.
///
/// Architecture:
///   * One MTL4CommandQueue, one MTL4Compiler, one render pipeline state
///     (built from default.metallib next to the bundle's binary).
///   * Double-buffered per-frame resources (allocator, frame uniforms,
///     part uniforms, argument table). A DispatchSemaphore throttles CPU
///     to at most 2 in-flight frames, signaled by an MTL4 commit-feedback.
///   * One shared vertex buffer + one shared index buffer holding every
///     mesh in the cat. Each part is a (vertexOffset, indexOffset,
///     indexCount) slice into those buffers.
///   * Render pass uses 4x MSAA color + depth, resolving to the CAMetalLayer
///     drawable.
final class MetalRenderer {

    // MARK: - Persistent objects

    let device: MTLDevice
    let layer:  CAMetalLayer

    private let queue:           MTL4CommandQueue
    private let compiler:        MTL4Compiler
    private let pipelineState:   MTLRenderPipelineState
    private let depthState:      MTLDepthStencilState
    private let starPipeline:    MTLRenderPipelineState
    private let starDepthState:  MTLDepthStencilState

    private let vertexBuffer:  MTLBuffer
    private let indexBuffer:   MTLBuffer

    // Metal 4 residency set: buffers bound via setAddress() (bindless) are
    // NOT automatically made resident. Without this, the GPU reads zeros and
    // every draw silently produces no fragments.
    private let residencySet: MTLResidencySet

    private let meshes: MeshLibrary
    private let scene:  CatScene

    // Pixel formats — Metal 4 declares these on the pipeline + render pass.
    private let colorFormat: MTLPixelFormat = .bgra8Unorm
    private let depthFormat: MTLPixelFormat = .depth32Float
    private let sampleCount: Int = 4

    // Resizable transient textures.
    private var msaaColor: MTLTexture?
    private var msaaDepth: MTLTexture?
    private var lastDrawableSize: CGSize = .zero

    // MARK: - Per-in-flight-frame resources

    private static let maxInFlight = 2
    private struct FrameResources {
        let allocator:      MTL4CommandAllocator
        let frameUniforms:  MTLBuffer
        let partUniforms:   MTLBuffer
        let argTable:       MTL4ArgumentTable
    }
    private var frames: [FrameResources] = []
    private var frameIdx: Int = 0
    private let semaphore = DispatchSemaphore(value: maxInFlight)

    // Each frame's partUniforms buffer holds up to this many parts in a row.
    // (The cat has ~40 leaf parts; round up for headroom.)
    private static let maxParts = 64

    private var lastFrameTime: CFTimeInterval = 0
    private var startTime: CFTimeInterval = 0

    // MARK: - Init

    init(layer: CAMetalLayer) throws {
        NSLog("CAT3D: Renderer.init start")
        guard let device = MTLCreateSystemDefaultDevice() else {
            NSLog("CAT3D: no Metal device")
            throw RendererError.noMetalDevice
        }
        NSLog("CAT3D: device = \(device.name), supportsFamily(apple7)=\(device.supportsFamily(.apple7))")
        self.device = device
        self.layer  = layer

        layer.device          = device
        layer.pixelFormat     = colorFormat
        layer.framebufferOnly = true
        layer.isOpaque        = true
        layer.maximumDrawableCount = 3
        layer.displaySyncEnabled   = true
        layer.allowsNextDrawableTimeout = false

        // Command queue
        NSLog("CAT3D: creating MTL4 command queue")
        guard let queue = device.makeMTL4CommandQueue() else {
            NSLog("CAT3D: MTL4 command queue creation returned nil (Metal 4 not supported on this hardware?)")
            throw RendererError.queueCreation
        }
        self.queue = queue
        NSLog("CAT3D: MTL4 queue ok")

        // Compiler
        NSLog("CAT3D: creating MTL4 compiler")
        let compilerDesc = MTL4CompilerDescriptor()
        compilerDesc.label = "CatCompiler"
        self.compiler = try device.makeCompiler(descriptor: compilerDesc)
        NSLog("CAT3D: MTL4 compiler ok")

        // Compile shader library at runtime via the Metal 4 compiler.
        NSLog("CAT3D: loading shader source + compiling library")
        let library = try Self.loadLibrary(compiler: compiler)
        NSLog("CAT3D: shader library ok; functions = \(library.functionNames)")

        // Build the render pipeline
        NSLog("CAT3D: building render pipeline state")
        self.pipelineState = try Self.makePipeline(
            compiler: compiler,
            library: library,
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            sampleCount: sampleCount
        )
        NSLog("CAT3D: pipeline state ok")

        // Depth-stencil state for the cat
        let dsDesc = MTLDepthStencilDescriptor()
        dsDesc.depthCompareFunction = .lessEqual
        dsDesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: dsDesc) else {
            NSLog("CAT3D: depth state creation failed")
            throw RendererError.depthStateCreation
        }
        self.depthState = ds

        // Depth-stencil state for the starfield: always pass, never write —
        // it paints the background and is overwritten by any opaque cat
        // fragment in front of it.
        let starDsDesc = MTLDepthStencilDescriptor()
        starDsDesc.depthCompareFunction = .always
        starDsDesc.isDepthWriteEnabled  = false
        guard let starDs = device.makeDepthStencilState(descriptor: starDsDesc) else {
            NSLog("CAT3D: star depth state creation failed")
            throw RendererError.depthStateCreation
        }
        self.starDepthState = starDs

        // Starfield pipeline (no vertex buffer; fullscreen triangle from vid).
        self.starPipeline = try Self.makeStarPipeline(
            compiler: compiler,
            library: library,
            colorFormat: colorFormat,
            sampleCount: sampleCount
        )
        NSLog("CAT3D: star pipeline ok")

        // Build meshes and upload to shared vertex/index buffers
        NSLog("CAT3D: building meshes")
        let (meshLib, vb, ib) = try Self.buildMeshes(device: device)
        self.meshes = meshLib
        self.vertexBuffer = vb
        self.indexBuffer  = ib
        NSLog("CAT3D: meshes ok (vb=\(vb.length)B, ib=\(ib.length)B)")

        // Build the cat scene graph
        self.scene = CatScene(mesh: meshLib)

        // Allocate per-frame resources
        NSLog("CAT3D: allocating per-frame resources")
        self.frames = try (0..<Self.maxInFlight).map { _ in
            try Self.makeFrameResources(device: device, maxParts: Self.maxParts)
        }

        // Build residency set covering every buffer we bind by GPU address.
        // Without this, Metal 4 bindless reads return zero and the cat is
        // invisible behind the clear color.
        let rsDesc = MTLResidencySetDescriptor()
        rsDesc.label = "CatResidencySet"
        rsDesc.initialCapacity = 4 + Self.maxInFlight * 2
        let rs = try device.makeResidencySet(descriptor: rsDesc)
        rs.addAllocation(vertexBuffer)
        rs.addAllocation(indexBuffer)
        for f in frames {
            rs.addAllocation(f.frameUniforms)
            rs.addAllocation(f.partUniforms)
        }
        rs.commit()
        queue.addResidencySet(rs)
        self.residencySet = rs

        startTime = CACurrentMediaTime()
        lastFrameTime = startTime
        NSLog("CAT3D: Renderer.init complete")
    }

    enum RendererError: Error {
        case noMetalDevice
        case queueCreation
        case libraryMissing
        case depthStateCreation
        case bufferAllocation
        case argumentTableCreation
        case noDrawable
    }

    // MARK: - Library + pipeline

    private static func loadLibrary(compiler: MTL4Compiler) throws -> MTLLibrary {
        // The build script copies Shaders.metal into the bundle's Resources
        // directory; we compile it at runtime via the Metal 4 compiler.
        let bundle = Bundle(for: MetalRenderer.self)
        guard let url = bundle.url(forResource: "Shaders", withExtension: "metal") else {
            throw RendererError.libraryMissing
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let desc = MTL4LibraryDescriptor()
        desc.source = source
        desc.name = "CatShaders"
        return try compiler.makeLibrary(descriptor: desc)
    }

    private static func makePipeline(compiler: MTL4Compiler,
                                     library: MTLLibrary,
                                     colorFormat: MTLPixelFormat,
                                     depthFormat: MTLPixelFormat,
                                     sampleCount: Int) throws -> MTLRenderPipelineState {
        let vd = MTL4LibraryFunctionDescriptor()
        vd.name = "vertex_main"
        vd.library = library

        let fd = MTL4LibraryFunctionDescriptor()
        fd.name = "fragment_main"
        fd.library = library

        let desc = MTL4RenderPipelineDescriptor()
        desc.label = "CatPipeline"
        desc.vertexFunctionDescriptor = vd
        desc.fragmentFunctionDescriptor = fd
        desc.colorAttachments[0].pixelFormat = colorFormat
        // In Metal 4 the depth/stencil format is on the render pass, not the
        // pipeline descriptor — Metal infers it at encode time.
        desc.rasterSampleCount = sampleCount

        return try compiler.makeRenderPipelineState(descriptor: desc,
                                                    compilerTaskOptions: nil)
    }

    private static func makeStarPipeline(compiler: MTL4Compiler,
                                         library: MTLLibrary,
                                         colorFormat: MTLPixelFormat,
                                         sampleCount: Int) throws -> MTLRenderPipelineState {
        let vd = MTL4LibraryFunctionDescriptor()
        vd.name = "vertex_starfield"
        vd.library = library

        let fd = MTL4LibraryFunctionDescriptor()
        fd.name = "fragment_starfield"
        fd.library = library

        let desc = MTL4RenderPipelineDescriptor()
        desc.label = "StarPipeline"
        desc.vertexFunctionDescriptor = vd
        desc.fragmentFunctionDescriptor = fd
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.rasterSampleCount = sampleCount

        return try compiler.makeRenderPipelineState(descriptor: desc,
                                                    compilerTaskOptions: nil)
    }

    // MARK: - Meshes

    private static func buildMeshes(device: MTLDevice) throws -> (MeshLibrary, MTLBuffer, MTLBuffer) {
        // Build all CPU meshes first.
        let body  = makeCapsule(radius: 0.46, height: 1.75, longitudeSegments: 48, latitudeSegments: 18)
        let belly = makeCapsule(radius: 0.36, height: 1.40, longitudeSegments: 40, latitudeSegments: 14)
        let stripe = makeTorus(ringRadius: 0.47, pipeRadius: 0.035, ringSegments: 48, pipeSegments: 14)
        let headSphere = makeSphere(radius: 0.48, longitudeSegments: 56, latitudeSegments: 36)
        let cheek = makeSphere(radius: 0.18, longitudeSegments: 32, latitudeSegments: 20)
        let muzzle = makeSphere(radius: 0.16, longitudeSegments: 32, latitudeSegments: 20)
        let earOuter = makeCone(topRadius: 0.012, bottomRadius: 0.17, height: 0.36)
        let earInner = makeCone(topRadius: 0.005, bottomRadius: 0.10, height: 0.24)
        let eyeWhite = makeSphere(radius: 0.11, longitudeSegments: 28, latitudeSegments: 18)
        let iris = makeSphere(radius: 0.085, longitudeSegments: 24, latitudeSegments: 16)
        let pupil = makeCapsule(radius: 0.018, height: 0.11, longitudeSegments: 16, latitudeSegments: 8)
        let nose = makeSphere(radius: 0.065, longitudeSegments: 24, latitudeSegments: 16)
        let mouth = makeBox(width: 0.025, height: 0.085, length: 0.025)
        let whisker = makeCylinder(radius: 0.005, height: 0.42, radialSegments: 12)
        let leg = makeCapsule(radius: 0.12, height: 0.65, longitudeSegments: 32, latitudeSegments: 12)
        let paw = makeSphere(radius: 0.15, longitudeSegments: 28, latitudeSegments: 18)
        let tail = makeCapsule(radius: 0.085, height: 1.05, longitudeSegments: 32, latitudeSegments: 12)
        let tailTip = makeSphere(radius: 0.11, longitudeSegments: 28, latitudeSegments: 18)

        let all: [MeshCPU] = [
            body, belly, stripe, headSphere, cheek, muzzle, earOuter, earInner,
            eyeWhite, iris, pupil, nose, mouth, whisker, leg, paw, tail, tailTip
        ]

        // Pack into one big vertex / index buffer.
        var slices: [MeshGPU] = []
        slices.reserveCapacity(all.count)

        var totalVerts = 0
        var totalInds  = 0
        for m in all {
            totalVerts += m.vertices.count
            totalInds  += m.indices.count
        }

        let vSize = totalVerts * MemoryLayout<CatVertex>.stride
        let iSize = totalInds  * MemoryLayout<UInt32>.stride

        guard let vBuf = device.makeBuffer(length: max(vSize, 64), options: .storageModeShared),
              let iBuf = device.makeBuffer(length: max(iSize, 64), options: .storageModeShared)
        else {
            throw RendererError.bufferAllocation
        }
        vBuf.label = "CatVertices"
        iBuf.label = "CatIndices"

        let vPtr = vBuf.contents().bindMemory(to: CatVertex.self, capacity: totalVerts)
        let iPtr = iBuf.contents().bindMemory(to: UInt32.self,    capacity: totalInds)

        var vCursor = 0
        var iCursor = 0
        for m in all {
            let vByteOffset = vCursor * MemoryLayout<CatVertex>.stride
            let iByteOffset = iCursor * MemoryLayout<UInt32>.stride

            for (k, v) in m.vertices.enumerated() {
                vPtr[vCursor + k] = CatVertex(
                    position: SIMD4(v.position.x, v.position.y, v.position.z, 1),
                    normal:   SIMD4(v.normal.x,   v.normal.y,   v.normal.z,   0),
                    uv:       SIMD4(v.uv.x,       v.uv.y,       0,             0)
                )
            }
            for (k, idx) in m.indices.enumerated() {
                iPtr[iCursor + k] = idx
            }

            slices.append(MeshGPU(vertexOffset: vByteOffset,
                                  indexOffset: iByteOffset,
                                  indexCount: m.indices.count))
            vCursor += m.vertices.count
            iCursor += m.indices.count
        }

        let lib = MeshLibrary(
            bodyCapsule:  slices[0],
            bellyCapsule: slices[1],
            stripeTorus:  slices[2],
            headSphere:   slices[3],
            cheekSphere:  slices[4],
            muzzleSphere: slices[5],
            earOuterCone: slices[6],
            earInnerCone: slices[7],
            eyeWhite:     slices[8],
            iris:         slices[9],
            pupilCapsule: slices[10],
            noseSphere:   slices[11],
            mouthBox:     slices[12],
            whiskerCyl:   slices[13],
            legCapsule:   slices[14],
            pawSphere:    slices[15],
            tailCapsule:  slices[16],
            tailTip:      slices[17]
        )
        return (lib, vBuf, iBuf)
    }

    // MARK: - Frame resources

    private static func makeFrameResources(device: MTLDevice,
                                           maxParts: Int) throws -> FrameResources {
        let allocatorDesc = MTL4CommandAllocatorDescriptor()
        allocatorDesc.label = "CatAllocator"
        let allocator = try device.makeCommandAllocator(descriptor: allocatorDesc)

        guard let frameUniforms = device.makeBuffer(length: MemoryLayout<FrameUniforms>.stride,
                                                    options: .storageModeShared),
              let partUniforms  = device.makeBuffer(length: MemoryLayout<PartUniforms>.stride * maxParts,
                                                    options: .storageModeShared)
        else {
            throw RendererError.bufferAllocation
        }
        frameUniforms.label = "CatFrameUniforms"
        partUniforms.label  = "CatPartUniforms"

        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.label = "CatArgTable"
        tableDesc.maxBufferBindCount = 8
        tableDesc.initializeBindings = true
        let argTable = try device.makeArgumentTable(descriptor: tableDesc)

        return FrameResources(allocator: allocator,
                              frameUniforms: frameUniforms,
                              partUniforms: partUniforms,
                              argTable: argTable)
    }

    // MARK: - Resize

    private func ensureMSAATextures(drawableSize: CGSize) {
        if drawableSize == lastDrawableSize, msaaColor != nil, msaaDepth != nil { return }
        lastDrawableSize = drawableSize
        let w = Int(max(drawableSize.width,  1))
        let h = Int(max(drawableSize.height, 1))

        let cDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: colorFormat, width: w, height: h, mipmapped: false)
        cDesc.textureType = .type2DMultisample
        cDesc.sampleCount = sampleCount
        cDesc.usage = [.renderTarget]
        cDesc.storageMode = .private

        let dDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: depthFormat, width: w, height: h, mipmapped: false)
        dDesc.textureType = .type2DMultisample
        dDesc.sampleCount = sampleCount
        dDesc.usage = [.renderTarget]
        dDesc.storageMode = .private

        msaaColor = device.makeTexture(descriptor: cDesc)
        msaaDepth = device.makeTexture(descriptor: dDesc)
        msaaColor?.label = "CatMSAAColor"
        msaaDepth?.label = "CatMSAADepth"
    }

    // MARK: - Frame

    private var loggedFirstFrame = false
    private var noDrawableCount = 0

    /// Advance and render exactly one frame. Safe to call from the main thread.
    func render() {
        if !loggedFirstFrame {
            NSLog("CAT3D: render() called for the first time; layer.drawableSize=\(layer.drawableSize)")
        }
        guard let drawable = layer.nextDrawable() else {
            noDrawableCount += 1
            if noDrawableCount == 1 || noDrawableCount % 60 == 0 {
                NSLog("CAT3D: nextDrawable() returned nil (count=\(noDrawableCount), drawableSize=\(layer.drawableSize), device=\(layer.device != nil))")
            }
            return
        }

        let drawableSize = layer.drawableSize
        ensureMSAATextures(drawableSize: drawableSize)
        guard let msaaColor = msaaColor, let msaaDepth = msaaDepth else {
            NSLog("CAT3D: MSAA textures nil (drawableSize=\(drawableSize))")
            return
        }
        if !loggedFirstFrame {
            NSLog("CAT3D: drawable + MSAA textures ok")
        }

        // Camera setup
        let aspect = Float(drawableSize.width / max(drawableSize.height, 1))
        let fovY: Float = radians(50)
        let nearZ: Float = 0.1
        let farZ:  Float = 200
        let cameraDistance: Float = 9
        let cameraPos = SIMD3<Float>(0, 0, cameraDistance)
        let view = matrixLookAt(eye: cameraPos, center: .zero, up: SIMD3(0, 1, 0))
        let proj = matrixPerspective(fovYRadians: fovY, aspect: aspect, near: nearZ, far: farZ)
        let viewProj = proj * view

        // Time + animation
        let now = CACurrentMediaTime()
        let dt = max(min(Float(now - lastFrameTime), 1.0 / 30), 1.0 / 240)
        lastFrameTime = now
        let elapsed = Float(now - startTime)

        // Visible world bounds at z=0 for screen-bouncing.
        let visibleH = 2 * tan(fovY * 0.5) * cameraDistance
        let visibleW = visibleH * max(aspect, 0.5)

        scene.update(dt: dt,
                     worldHalfWidth:  visibleW * 0.5,
                     worldHalfHeight: visibleH * 0.5)

        // Wait for an in-flight slot.
        semaphore.wait()
        let frame = frames[frameIdx]
        frameIdx = (frameIdx + 1) % Self.maxInFlight

        // Reset the allocator now that we know the previous frame that used it
        // has completed (the semaphore guarantees that).
        frame.allocator.reset()

        // Write frame uniforms.
        var fu = FrameUniforms()
        fu.viewProjection = viewProj
        fu.cameraPosTime = SIMD4(cameraPos, elapsed)
        fu.hueParams = SIMD4(0.065, 0, 0, 0)
        fu.keyLightDir    = SIMD4(simd_normalize(SIMD3<Float>(0.6, 0.9, 0.4)), 0)
        fu.keyLightColor  = SIMD4(1.0, 1.0, 1.0, 0)
        fu.fillLightDir   = SIMD4(simd_normalize(SIMD3<Float>(-0.5, -0.3, 0.8)), 0)
        fu.fillLightColor = SIMD4(0.55, 0.65, 1.0, 0) * 0.6
        fu.ambientColor   = SIMD4(0.40, 0.40, 0.42, 0)
        frame.frameUniforms.contents().assumingMemoryBound(to: FrameUniforms.self).pointee = fu

        // Write part uniforms.
        let items = scene.gatherDrawItems()
        let partCount = min(items.count, Self.maxParts)
        let partsPtr = frame.partUniforms.contents().assumingMemoryBound(to: PartUniforms.self)
        for i in 0..<partCount { partsPtr[i] = items[i].uniforms }

        // Build the render pass.
        let pass = MTL4RenderPassDescriptor()
        pass.colorAttachments[0].texture        = msaaColor
        pass.colorAttachments[0].resolveTexture = drawable.texture
        pass.colorAttachments[0].loadAction     = .clear
        pass.colorAttachments[0].storeAction    = .multisampleResolve
        // Black clear — the starfield fragment paints the actual background.
        pass.colorAttachments[0].clearColor     = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.depthAttachment.texture            = msaaDepth
        pass.depthAttachment.loadAction         = .clear
        pass.depthAttachment.storeAction        = .dontCare
        pass.depthAttachment.clearDepth         = 1.0

        guard let cmd = device.makeCommandBuffer() else {
            NSLog("CAT3D: makeCommandBuffer() returned nil")
            semaphore.signal(); return
        }
        cmd.beginCommandBuffer(allocator: frame.allocator)
        cmd.useResidencySet(residencySet)

        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else {
            NSLog("CAT3D: makeRenderCommandEncoder returned nil")
            cmd.endCommandBuffer()
            semaphore.signal()
            return
        }
        if !loggedFirstFrame {
            NSLog("CAT3D: render encoder ok; partCount=\(partCount)")
        }
        enc.label = "CatPass"
        enc.setViewport(MTLViewport(originX: 0, originY: 0,
                                    width: drawableSize.width, height: drawableSize.height,
                                    znear: 0, zfar: 1))
        enc.setFrontFacing(.counterClockwise)
        enc.setCullMode(.back)
        enc.setArgumentTable(frame.argTable, stages: [.vertex, .fragment])

        // Frame uniforms (slot 1) are shared between the starfield and the cat,
        // so bind them once up front.
        frame.argTable.setAddress(frame.frameUniforms.gpuAddress,
                                  index: Int(CatBufferIndexFrame.rawValue))

        // ---- Starfield background ----
        enc.setRenderPipelineState(starPipeline)
        enc.setDepthStencilState(starDepthState)
        enc.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 3)

        // ---- Cat ----
        enc.setRenderPipelineState(pipelineState)
        enc.setDepthStencilState(depthState)
        frame.argTable.setAddress(vertexBuffer.gpuAddress,
                                  index: Int(CatBufferIndexVertices.rawValue))

        let partStride = MemoryLayout<PartUniforms>.stride
        let indexBase  = indexBuffer.gpuAddress
        let indexEnd   = indexBuffer.length

        for i in 0..<partCount {
            let item = items[i]
            // Rebind the per-part uniform slot to the right slice.
            frame.argTable.setAddress(frame.partUniforms.gpuAddress + UInt64(i * partStride),
                                      index: Int(CatBufferIndexPart.rawValue))

            // Vertex pull mode: we read CatVertex out of the vertex buffer via
            // vertex_id inside the shader, with a per-mesh base offset baked
            // into the argument table's vertex slot.
            frame.argTable.setAddress(vertexBuffer.gpuAddress + UInt64(item.mesh.vertexOffset),
                                      index: Int(CatBufferIndexVertices.rawValue))

            enc.drawIndexedPrimitives(primitiveType: .triangle,
                                       indexCount: item.mesh.indexCount,
                                       indexType: .uint32,
                                       indexBuffer: indexBase + UInt64(item.mesh.indexOffset),
                                       indexBufferLength: indexEnd - item.mesh.indexOffset,
                                       instanceCount: item.instanceCount)
        }

        enc.endEncoding()
        cmd.endCommandBuffer()

        // Schedule presentation and commit.
        queue.signalDrawable(drawable)
        drawable.present()

        let opts = MTL4CommitOptions()
        opts.addFeedbackHandler { [semaphore] _ in
            semaphore.signal()
        }
        queue.commit([cmd], options: opts)

        if !loggedFirstFrame {
            NSLog("CAT3D: first frame committed")
            loggedFirstFrame = true
        }
    }
}
