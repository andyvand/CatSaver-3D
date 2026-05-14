import Metal
import simd

struct Vertex {
    var position: SIMD3<Float>
    var normal:   SIMD3<Float>
    var uv:       SIMD2<Float>
}

/// CPU-side mesh, later uploaded into a single big vertex/index buffer.
struct MeshCPU {
    var vertices: [Vertex]
    var indices:  [UInt32]
}

/// GPU-side mesh slice — points at a region inside the renderer's shared
/// vertex/index buffers.
struct MeshGPU {
    var vertexOffset: Int   // byte offset into the shared vertex buffer
    var indexOffset:  Int   // byte offset into the shared index buffer
    var indexCount:   Int
}

// MARK: - Sphere

func makeSphere(radius: Float, longitudeSegments: Int = 48, latitudeSegments: Int = 32) -> MeshCPU {
    let lon = max(longitudeSegments, 3)
    let lat = max(latitudeSegments, 2)
    var verts: [Vertex] = []
    var inds: [UInt32] = []
    verts.reserveCapacity((lat + 1) * (lon + 1))
    inds.reserveCapacity(lat * lon * 6)

    for j in 0...lat {
        let vy = Float(j) / Float(lat)
        let theta = vy * .pi
        let sinT = sin(theta)
        let cosT = cos(theta)
        for i in 0...lon {
            let vx = Float(i) / Float(lon)
            let phi = vx * 2 * .pi
            let n = SIMD3<Float>(sinT * cos(phi), cosT, sinT * sin(phi))
            verts.append(Vertex(position: n * radius, normal: n, uv: SIMD2(vx, vy)))
        }
    }
    for j in 0..<lat {
        for i in 0..<lon {
            let a = UInt32(j * (lon + 1) + i)
            let b = UInt32(j * (lon + 1) + i + 1)
            let c = UInt32((j + 1) * (lon + 1) + i)
            let d = UInt32((j + 1) * (lon + 1) + i + 1)
            // CCW-from-outside winding. The j-loop runs north-pole to
            // south-pole, so y decreases as j increases — that flips the
            // winding relative to a y-up traversal, hence (a,b,c)+(b,d,c).
            inds.append(contentsOf: [a, b, c, b, d, c])
        }
    }
    return MeshCPU(vertices: verts, indices: inds)
}

// MARK: - Capsule (Y-axis aligned: total length = `height`, cap radius = `radius`)

func makeCapsule(radius: Float, height: Float,
                 longitudeSegments: Int = 48, latitudeSegments: Int = 16) -> MeshCPU {
    // Capsule = top hemisphere + cylinder + bottom hemisphere, all stitched.
    // We sweep latitude from 0 (top pole) to pi (bottom pole); when in the
    // top hemi we offset y by +cylHalf, when bottom by -cylHalf.
    let lon = max(longitudeSegments, 3)
    let lat = max(latitudeSegments, 4)
    let cylHalf = max((height - 2 * radius) * 0.5, 0)
    var verts: [Vertex] = []
    var inds: [UInt32] = []
    verts.reserveCapacity((lat + 1) * (lon + 1))
    inds.reserveCapacity(lat * lon * 6)

    for j in 0...lat {
        let vy = Float(j) / Float(lat)
        let theta = vy * .pi
        let sinT = sin(theta)
        let cosT = cos(theta)
        let yOffset: Float = (theta <= .pi * 0.5) ? cylHalf : -cylHalf
        for i in 0...lon {
            let vx = Float(i) / Float(lon)
            let phi = vx * 2 * .pi
            let n = SIMD3<Float>(sinT * cos(phi), cosT, sinT * sin(phi))
            let pos = SIMD3<Float>(n.x * radius, n.y * radius + yOffset, n.z * radius)
            verts.append(Vertex(position: pos, normal: n, uv: SIMD2(vx, vy)))
        }
    }
    for j in 0..<lat {
        for i in 0..<lon {
            let a = UInt32(j * (lon + 1) + i)
            let b = UInt32(j * (lon + 1) + i + 1)
            let c = UInt32((j + 1) * (lon + 1) + i)
            let d = UInt32((j + 1) * (lon + 1) + i + 1)
            // Same north-to-south traversal as the sphere — see comment there.
            inds.append(contentsOf: [a, b, c, b, d, c])
        }
    }
    return MeshCPU(vertices: verts, indices: inds)
}

// MARK: - Cone (Y-axis aligned: bottom at -h/2, top at +h/2)

func makeCone(topRadius: Float, bottomRadius: Float, height: Float,
              radialSegments: Int = 36, heightSegments: Int = 1) -> MeshCPU {
    let rs = max(radialSegments, 3)
    let hs = max(heightSegments, 1)
    var verts: [Vertex] = []
    var inds: [UInt32] = []

    // Side surface
    let slant = SIMD2<Float>(bottomRadius - topRadius, height)
    let slantLen = simd_length(slant)
    let slantDir = slant / max(slantLen, 1e-6)
    // For a cone, the side normal points outward perpendicular to the slant.
    // In the (r, y) plane the slant direction is (slantDir.x, slantDir.y);
    // perpendicular outward is (slantDir.y, -slantDir.x).
    let nR = slantDir.y
    let nY = -slantDir.x

    for j in 0...hs {
        let vy = Float(j) / Float(hs)
        let y  = -height * 0.5 + height * vy
        let r  = bottomRadius + (topRadius - bottomRadius) * vy
        for i in 0...rs {
            let vx = Float(i) / Float(rs)
            let phi = vx * 2 * .pi
            let cx = cos(phi)
            let sz = sin(phi)
            let pos = SIMD3<Float>(cx * r, y, sz * r)
            let n   = simd_normalize(SIMD3<Float>(cx * nR, nY, sz * nR))
            verts.append(Vertex(position: pos, normal: n, uv: SIMD2(vx, vy)))
        }
    }
    for j in 0..<hs {
        for i in 0..<rs {
            let a = UInt32(j * (rs + 1) + i)
            let b = UInt32(j * (rs + 1) + i + 1)
            let c = UInt32((j + 1) * (rs + 1) + i)
            let d = UInt32((j + 1) * (rs + 1) + i + 1)
            inds.append(contentsOf: [a, c, b, b, c, d])
        }
    }

    // Bottom cap (centered fan)
    if bottomRadius > 0 {
        let centerIdx = UInt32(verts.count)
        verts.append(Vertex(position: SIMD3(0, -height * 0.5, 0),
                            normal:   SIMD3(0, -1, 0),
                            uv:       SIMD2(0.5, 0.5)))
        let ringStart = UInt32(verts.count)
        for i in 0...rs {
            let phi = Float(i) / Float(rs) * 2 * .pi
            let cx = cos(phi); let sz = sin(phi)
            verts.append(Vertex(position: SIMD3(cx * bottomRadius, -height * 0.5, sz * bottomRadius),
                                normal:   SIMD3(0, -1, 0),
                                uv:       SIMD2(cx * 0.5 + 0.5, sz * 0.5 + 0.5)))
        }
        // Outward normal is -y; CCW-from-outside means CCW when viewed from
        // below the cap looking up.
        for i in 0..<rs {
            inds.append(contentsOf: [centerIdx, ringStart + UInt32(i), ringStart + UInt32(i + 1)])
        }
    }

    // Top cap (only if topRadius > 0)
    if topRadius > 0 {
        let centerIdx = UInt32(verts.count)
        verts.append(Vertex(position: SIMD3(0, height * 0.5, 0),
                            normal:   SIMD3(0, 1, 0),
                            uv:       SIMD2(0.5, 0.5)))
        let ringStart = UInt32(verts.count)
        for i in 0...rs {
            let phi = Float(i) / Float(rs) * 2 * .pi
            let cx = cos(phi); let sz = sin(phi)
            verts.append(Vertex(position: SIMD3(cx * topRadius, height * 0.5, sz * topRadius),
                                normal:   SIMD3(0, 1, 0),
                                uv:       SIMD2(cx * 0.5 + 0.5, sz * 0.5 + 0.5)))
        }
        // Outward normal is +y; CCW-from-outside means CCW when viewed from
        // above the cap looking down.
        for i in 0..<rs {
            inds.append(contentsOf: [centerIdx, ringStart + UInt32(i + 1), ringStart + UInt32(i)])
        }
    }
    return MeshCPU(vertices: verts, indices: inds)
}

// MARK: - Cylinder (Y-axis aligned)

func makeCylinder(radius: Float, height: Float,
                  radialSegments: Int = 24, heightSegments: Int = 1) -> MeshCPU {
    return makeCone(topRadius: radius, bottomRadius: radius, height: height,
                    radialSegments: radialSegments, heightSegments: heightSegments)
}

// MARK: - Torus (Y-axis up; ring lies in the XZ plane)

func makeTorus(ringRadius: Float, pipeRadius: Float,
               ringSegments: Int = 48, pipeSegments: Int = 14) -> MeshCPU {
    let rs = max(ringSegments, 3)
    let ps = max(pipeSegments, 3)
    var verts: [Vertex] = []
    var inds: [UInt32] = []
    for j in 0...ps {
        let vy = Float(j) / Float(ps)
        let v = vy * 2 * .pi
        let cv = cos(v); let sv = sin(v)
        for i in 0...rs {
            let vx = Float(i) / Float(rs)
            let u = vx * 2 * .pi
            let cu = cos(u); let su = sin(u)
            let x = (ringRadius + pipeRadius * cv) * cu
            let z = (ringRadius + pipeRadius * cv) * su
            let y = pipeRadius * sv
            let n = SIMD3<Float>(cv * cu, sv, cv * su)
            verts.append(Vertex(position: SIMD3(x, y, z), normal: n, uv: SIMD2(vx, vy)))
        }
    }
    for j in 0..<ps {
        for i in 0..<rs {
            let a = UInt32(j * (rs + 1) + i)
            let b = UInt32(j * (rs + 1) + i + 1)
            let c = UInt32((j + 1) * (rs + 1) + i)
            let d = UInt32((j + 1) * (rs + 1) + i + 1)
            inds.append(contentsOf: [a, c, b, b, c, d])
        }
    }
    return MeshCPU(vertices: verts, indices: inds)
}

// MARK: - Box

func makeBox(width: Float, height: Float, length: Float) -> MeshCPU {
    let hx = width  * 0.5
    let hy = height * 0.5
    let hz = length * 0.5
    let faces: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = [
        // (v0, v1, v2, v3, normal)  CCW winding when viewed from outside
        (SIMD3(-hx,-hy, hz), SIMD3( hx,-hy, hz), SIMD3( hx, hy, hz), SIMD3(-hx, hy, hz), SIMD3(0,0, 1)),  // +Z
        (SIMD3( hx,-hy,-hz), SIMD3(-hx,-hy,-hz), SIMD3(-hx, hy,-hz), SIMD3( hx, hy,-hz), SIMD3(0,0,-1)),  // -Z
        (SIMD3( hx,-hy, hz), SIMD3( hx,-hy,-hz), SIMD3( hx, hy,-hz), SIMD3( hx, hy, hz), SIMD3( 1,0,0)),  // +X
        (SIMD3(-hx,-hy,-hz), SIMD3(-hx,-hy, hz), SIMD3(-hx, hy, hz), SIMD3(-hx, hy,-hz), SIMD3(-1,0,0)),  // -X
        (SIMD3(-hx, hy, hz), SIMD3( hx, hy, hz), SIMD3( hx, hy,-hz), SIMD3(-hx, hy,-hz), SIMD3(0, 1,0)),  // +Y
        (SIMD3(-hx,-hy,-hz), SIMD3( hx,-hy,-hz), SIMD3( hx,-hy, hz), SIMD3(-hx,-hy, hz), SIMD3(0,-1,0)),  // -Y
    ]
    var verts: [Vertex] = []
    var inds: [UInt32] = []
    for (v0, v1, v2, v3, n) in faces {
        let base = UInt32(verts.count)
        verts.append(Vertex(position: v0, normal: n, uv: SIMD2(0, 0)))
        verts.append(Vertex(position: v1, normal: n, uv: SIMD2(1, 0)))
        verts.append(Vertex(position: v2, normal: n, uv: SIMD2(1, 1)))
        verts.append(Vertex(position: v3, normal: n, uv: SIMD2(0, 1)))
        inds.append(contentsOf: [base, base+1, base+2, base, base+2, base+3])
    }
    return MeshCPU(vertices: verts, indices: inds)
}
