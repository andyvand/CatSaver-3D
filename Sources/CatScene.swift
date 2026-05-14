import Metal
import simd

/// A node in the cat's transform hierarchy. Pivot nodes (no mesh) drive
/// animation; leaf nodes carry a mesh + per-part shader uniforms.
final class CatNode {
    var localTransform: simd_float4x4 = matrixIdentity()
    var worldTransform: simd_float4x4 = matrixIdentity()
    weak var parent: CatNode?
    var children: [CatNode] = []

    var mesh: MeshGPU?
    var baseColor: SIMD3<Float> = .one
    var hueOffset: Float = 0
    var tint: Float = 1
    var doFur: Bool = false
    var shellCount: Int = 1
    var furLength: Float = 0

    func addChild(_ c: CatNode) {
        c.parent = self
        children.append(c)
    }

    func updateWorld() {
        if let parent = parent {
            worldTransform = parent.worldTransform * localTransform
        } else {
            worldTransform = localTransform
        }
        for c in children { c.updateWorld() }
    }
}

/// One ready-to-draw entry for the renderer.
struct DrawItem {
    var mesh: MeshGPU
    var uniforms: PartUniforms
    var instanceCount: Int
}

/// All the meshes the cat uses. Built and uploaded once.
struct MeshLibrary {
    let bodyCapsule:  MeshGPU
    let bellyCapsule: MeshGPU
    let stripeTorus:  MeshGPU
    let headSphere:   MeshGPU
    let cheekSphere:  MeshGPU
    let muzzleSphere: MeshGPU
    let earOuterCone: MeshGPU
    let earInnerCone: MeshGPU
    let eyeWhite:     MeshGPU
    let iris:         MeshGPU
    let pupilCapsule: MeshGPU
    let noseSphere:   MeshGPU
    let mouthBox:     MeshGPU
    let whiskerCyl:   MeshGPU
    let legCapsule:   MeshGPU
    let pawSphere:    MeshGPU
    let tailCapsule:  MeshGPU
    let tailTip:      MeshGPU
}

final class CatScene {
    // Standard fur knobs.
    static let furShells: Int    = 22
    static let furLength: Float  = 0.075

    // Persistent palette (gets re-tinted by the rainbow shader at run time).
    static let furColor    = SIMD3<Float>(0.95, 0.72, 0.42)
    static let bellyColor  = SIMD3<Float>(0.99, 0.88, 0.70)
    static let stripeColor = SIMD3<Float>(0.62, 0.40, 0.18)
    static let pinkColor   = SIMD3<Float>(1.00, 0.62, 0.66)
    static let darkColor   = SIMD3<Float>(0.08, 0.06, 0.04)
    static let eyeColor    = SIMD3<Float>(0.45, 0.85, 0.50)
    static let whiteColor  = SIMD3<Float>(1, 1, 1)

    let root: CatNode = CatNode()

    // Pivots that get animated each frame.
    private let bodyContainer = CatNode()
    private let bellyContainer = CatNode()
    private let head           = CatNode()
    private let leftEar        = CatNode()
    private let rightEar       = CatNode()
    private let tailPivot      = CatNode()

    // Animation state.
    var time: Float = 0
    var bouncePos = SIMD2<Float>(0, 0)
    var bounceVel = SIMD2<Float>(2.7, 1.92)   // world units per second

    private var leftEarLastTwitchAt:  Float = -10
    private var leftEarNextTwitchAt:  Float = 1.2
    private var rightEarLastTwitchAt: Float = -10
    private var rightEarNextTwitchAt: Float = 2.6

    private let mesh: MeshLibrary

    init(mesh: MeshLibrary) {
        self.mesh = mesh
        buildHierarchy()
    }

    // MARK: - Hierarchy

    private func buildHierarchy() {
        // ---- Body capsule (rotated so the long axis runs along X) ----
        bodyContainer.localTransform = matrixRotateZ(.pi * 0.5)
        attachFur(bodyContainer, mesh: mesh.bodyCapsule,
                  color: Self.furColor, hueOffset: 0.0)
        root.addChild(bodyContainer)

        // ---- Belly accent ----
        bellyContainer.localTransform =
            matrixTranslate(SIMD3(0, -0.18, 0.05)) * matrixRotateZ(.pi * 0.5)
        attachFur(bellyContainer, mesh: mesh.bellyCapsule,
                  color: Self.bellyColor, shells: 18, furLen: 0.05, hueOffset: 0.12)
        root.addChild(bellyContainer)

        // ---- Tabby stripes ----
        let stripeX: [Float] = [-0.55, -0.27, 0.01, 0.29, 0.55]
        for x in stripeX {
            let n = CatNode()
            n.localTransform = matrixTranslate(SIMD3(x, 0, 0)) * matrixRotateZ(.pi * 0.5)
            attachFur(n, mesh: mesh.stripeTorus,
                      color: Self.stripeColor, shells: 10, furLen: 0.028, hueOffset: 0.55)
            root.addChild(n)
        }

        // ---- Head ----
        head.localTransform = matrixTranslate(SIMD3(0.98, 0.22, 0))
        attachFur(head, mesh: mesh.headSphere, color: Self.furColor, hueOffset: 0.0)
        root.addChild(head)

        // Cheeks
        for z: Float in [0.28, -0.28] {
            let c = CatNode()
            c.localTransform = matrixTranslate(SIMD3(0.32, -0.08, z))
            attachFur(c, mesh: mesh.cheekSphere,
                      color: Self.furColor, shells: 16, furLen: 0.055, hueOffset: 0.05)
            head.addChild(c)
        }

        // Muzzle
        let muzzle = CatNode()
        muzzle.localTransform = matrixTranslate(SIMD3(0.42, -0.13, 0))
        attachFur(muzzle, mesh: mesh.muzzleSphere,
                  color: Self.bellyColor, shells: 14, furLen: 0.045, hueOffset: 0.12)
        head.addChild(muzzle)

        // Ears (with twitch pivots)
        configureEar(pivot: leftEar,  side:  1)
        configureEar(pivot: rightEar, side: -1)
        head.addChild(leftEar)
        head.addChild(rightEar)

        // Eyes (smooth)
        for z: Float in [0.19, -0.19] {
            let eye = CatNode()
            eye.localTransform = matrixTranslate(SIMD3(0.38, 0.10, z))
            attachSmooth(eye, mesh: mesh.eyeWhite,
                         color: Self.whiteColor, hueOffset: 0.7, tint: 0.55)
            head.addChild(eye)

            let iris = CatNode()
            iris.localTransform = matrixTranslate(SIMD3(0.04, 0, 0))
            attachSmooth(iris, mesh: mesh.iris,
                         color: Self.eyeColor, hueOffset: 0.33, tint: 1.0)
            eye.addChild(iris)

            let pupil = CatNode()
            pupil.localTransform = matrixTranslate(SIMD3(0.05, 0, 0))
            attachSmooth(pupil, mesh: mesh.pupilCapsule,
                         color: Self.darkColor, hueOffset: 0.0, tint: 0.0)
            eye.addChild(pupil)
        }

        // Nose
        let nose = CatNode()
        nose.localTransform = matrixTranslate(SIMD3(0.49, -0.06, 0))
        attachSmooth(nose, mesh: mesh.noseSphere,
                     color: Self.pinkColor, hueOffset: 0.85, tint: 1.0)
        head.addChild(nose)

        // Mouth
        let mouth = CatNode()
        mouth.localTransform = matrixTranslate(SIMD3(0.48, -0.20, 0))
        attachSmooth(mouth, mesh: mesh.mouthBox,
                     color: Self.darkColor, hueOffset: 0.0, tint: 0.0)
        head.addChild(mouth)

        // Whiskers
        for side: Float in [1, -1] {
            for tilt: Float in [-0.18, 0.0, 0.18] {
                let w = CatNode()
                let rotZ: Float = side > 0 ? .pi * 0.5 : -.pi * 0.5
                w.localTransform =
                    matrixTranslate(SIMD3(0.42, -0.06, 0.18 * side)) *
                    matrixRotateZ(rotZ) *
                    matrixRotateX(tilt)
                attachSmooth(w, mesh: mesh.whiskerCyl,
                             color: Self.whiteColor, hueOffset: 0.7, tint: 0.5)
                head.addChild(w)
            }
        }

        // Legs + paws
        let legPositions: [(Float, Float)] = [
            ( 0.58,  0.28), ( 0.58, -0.28),
            (-0.58,  0.28), (-0.58, -0.28),
        ]
        for (x, z) in legPositions {
            let leg = CatNode()
            leg.localTransform = matrixTranslate(SIMD3(x, -0.48, z))
            attachFur(leg, mesh: mesh.legCapsule,
                      color: Self.furColor, shells: 16, furLen: 0.05, hueOffset: 0.0)
            root.addChild(leg)

            let paw = CatNode()
            paw.localTransform = matrixTranslate(SIMD3(x, -0.78, z))
            attachFur(paw, mesh: mesh.pawSphere,
                      color: Self.stripeColor, shells: 12, furLen: 0.04, hueOffset: 0.55)
            root.addChild(paw)
        }

        // Tail
        tailPivot.localTransform = matrixTranslate(SIMD3(-0.92, 0.28, 0))
        root.addChild(tailPivot)

        let tail = CatNode()
        tail.localTransform = matrixTranslate(SIMD3(0, 0.45, 0)) * matrixRotateZ(.pi / 5)
        attachFur(tail, mesh: mesh.tailCapsule,
                  color: Self.furColor, shells: 18, furLen: 0.06, hueOffset: 0.0)
        tailPivot.addChild(tail)

        let tailTipNode = CatNode()
        tailTipNode.localTransform = matrixTranslate(SIMD3(-0.36, 0.88, 0))
        attachFur(tailTipNode, mesh: mesh.tailTip,
                  color: Self.stripeColor, shells: 14, furLen: 0.05, hueOffset: 0.55)
        tailPivot.addChild(tailTipNode)
    }

    private func configureEar(pivot: CatNode, side: Float) {
        pivot.localTransform = matrixTranslate(SIMD3(-0.05, 0.38, 0.24 * side))

        let outer = CatNode()
        outer.localTransform =
            matrixTranslate(SIMD3(0, 0.18, 0)) * matrixRotateZ(-0.18 * side)
        attachFur(outer, mesh: mesh.earOuterCone,
                  color: Self.furColor, shells: 14, furLen: 0.05, hueOffset: 0.0)
        pivot.addChild(outer)

        let inner = CatNode()
        inner.localTransform =
            matrixTranslate(SIMD3(0.03, 0.13, 0)) * matrixRotateZ(-0.18 * side)
        attachSmooth(inner, mesh: mesh.earInnerCone,
                     color: Self.pinkColor, hueOffset: 0.85, tint: 1.0)
        pivot.addChild(inner)
    }

    private func attachFur(_ node: CatNode, mesh: MeshGPU, color: SIMD3<Float>,
                           shells: Int = furShells, furLen: Float = furLength,
                           hueOffset: Float = 0, tint: Float = 1.0) {
        node.mesh = mesh
        node.baseColor = color
        node.shellCount = shells
        node.furLength = furLen
        node.hueOffset = hueOffset
        node.tint = tint
        node.doFur = true
    }

    private func attachSmooth(_ node: CatNode, mesh: MeshGPU, color: SIMD3<Float>,
                              hueOffset: Float = 0, tint: Float = 1.0) {
        node.mesh = mesh
        node.baseColor = color
        node.shellCount = 1
        node.furLength = 0
        node.hueOffset = hueOffset
        node.tint = tint
        node.doFur = false
    }

    // MARK: - Per-frame update

    /// Advance animation and bouncing-around-the-screen movement.
    /// `worldHalfWidth` / `worldHalfHeight` come from the camera setup.
    func update(dt: Float, worldHalfWidth: Float, worldHalfHeight: Float) {
        time += dt

        // Bouncing
        bouncePos += bounceVel * dt
        let margin: Float = 1.4
        let xLimit = worldHalfWidth  - margin
        let yLimit = worldHalfHeight - margin
        if bouncePos.x >  xLimit { bouncePos.x =  xLimit; bounceVel.x = -abs(bounceVel.x) }
        if bouncePos.x < -xLimit { bouncePos.x = -xLimit; bounceVel.x =  abs(bounceVel.x) }
        if bouncePos.y >  yLimit { bouncePos.y =  yLimit; bounceVel.y = -abs(bounceVel.y) }
        if bouncePos.y < -yLimit { bouncePos.y = -yLimit; bounceVel.y =  abs(bounceVel.y) }

        // Root: bounce position + slow Y-axis spin
        let spinY = time * (2 * .pi / 7.5)
        root.localTransform =
            matrixTranslate(SIMD3(bouncePos.x, bouncePos.y, 0)) *
            matrixRotateY(spinY)

        // Body & head bob (synchronized vertical sin)
        let bob = sin(time * 7.85) * 0.06
        bodyContainer.localTransform =
            matrixTranslate(SIMD3(0, bob, 0)) * matrixRotateZ(.pi * 0.5)
        head.localTransform =
            matrixTranslate(SIMD3(0.98, 0.22 + bob, 0))

        // Tail wag
        let wag = sin(time * (2 * .pi / 1.8)) * 0.45
        tailPivot.localTransform =
            matrixTranslate(SIMD3(-0.92, 0.28, 0)) * matrixRotateZ(wag)

        // Ear twitches
        updateEar(&leftEarLastTwitchAt,  &leftEarNextTwitchAt,  pivot: leftEar,  base: SIMD3(-0.05, 0.38,  0.24))
        updateEar(&rightEarLastTwitchAt, &rightEarNextTwitchAt, pivot: rightEar, base: SIMD3(-0.05, 0.38, -0.24))

        root.updateWorld()
    }

    private func updateEar(_ last: inout Float, _ next: inout Float,
                           pivot: CatNode, base: SIMD3<Float>) {
        let twitchDur: Float = 0.18
        let phase = (time - last) / twitchDur
        var angle: Float = 0
        if phase < 1 {
            angle = sin(phase * .pi) * 0.28
        } else if time > next {
            last = time
            next = time + Float.random(in: 1.8 ... 4.5) + twitchDur
        }
        pivot.localTransform = matrixTranslate(base) * matrixRotateZ(angle)
    }

    // MARK: - Draw items

    /// Collect the renderable parts into a flat draw list, with model + normal
    /// matrices and per-part shader uniforms ready to ship to the GPU.
    func gatherDrawItems() -> [DrawItem] {
        var items: [DrawItem] = []
        items.reserveCapacity(32)
        gather(node: root, into: &items)
        return items
    }

    private func gather(node: CatNode, into out: inout [DrawItem]) {
        if let mesh = node.mesh {
            let model = node.worldTransform
            var u = PartUniforms()
            u.model = model
            u.normalMatrix = normalMatrix(of: model)
            u.baseColor = SIMD4(node.baseColor, 1)
            u.furParams = SIMD4(Float(node.shellCount), node.furLength, node.hueOffset, node.tint)
            u.miscParams = SIMD4(node.doFur ? 1 : 0, 0, 0, 0)
            out.append(DrawItem(mesh: mesh, uniforms: u, instanceCount: max(node.shellCount, 1)))
        }
        for c in node.children {
            gather(node: c, into: &out)
        }
    }
}
