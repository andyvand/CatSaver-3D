import simd

@inline(__always) func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

func matrixIdentity() -> simd_float4x4 { matrix_identity_float4x4 }

func matrixTranslate(_ t: SIMD3<Float>) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.3 = SIMD4(t.x, t.y, t.z, 1)
    return m
}

func matrixScale(_ s: SIMD3<Float>) -> simd_float4x4 {
    simd_float4x4(diagonal: SIMD4(s.x, s.y, s.z, 1))
}

func matrixUniformScale(_ s: Float) -> simd_float4x4 {
    simd_float4x4(diagonal: SIMD4(s, s, s, 1))
}

func matrixRotateX(_ a: Float) -> simd_float4x4 {
    let c = cos(a), s = sin(a)
    return simd_float4x4(
        SIMD4(1, 0, 0, 0),
        SIMD4(0, c, s, 0),
        SIMD4(0, -s, c, 0),
        SIMD4(0, 0, 0, 1)
    )
}

func matrixRotateY(_ a: Float) -> simd_float4x4 {
    let c = cos(a), s = sin(a)
    return simd_float4x4(
        SIMD4(c, 0, -s, 0),
        SIMD4(0, 1,  0, 0),
        SIMD4(s, 0,  c, 0),
        SIMD4(0, 0,  0, 1)
    )
}

func matrixRotateZ(_ a: Float) -> simd_float4x4 {
    let c = cos(a), s = sin(a)
    return simd_float4x4(
        SIMD4( c, s, 0, 0),
        SIMD4(-s, c, 0, 0),
        SIMD4( 0, 0, 1, 0),
        SIMD4( 0, 0, 0, 1)
    )
}

/// Right-handed perspective matrix mapping NDC z to [0, 1] (Metal convention).
func matrixPerspective(fovYRadians fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let ys = 1 / tan(fovY * 0.5)
    let xs = ys / aspect
    let zs = far / (near - far)
    return simd_float4x4(
        SIMD4(xs, 0,  0,           0),
        SIMD4(0,  ys, 0,           0),
        SIMD4(0,  0,  zs,         -1),
        SIMD4(0,  0,  zs * near,   0)
    )
}

/// Right-handed lookAt view matrix.
func matrixLookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    let f = normalize(center - eye)
    let s = normalize(cross(f, up))
    let u = cross(s, f)
    return simd_float4x4(
        SIMD4(s.x, u.x, -f.x, 0),
        SIMD4(s.y, u.y, -f.y, 0),
        SIMD4(s.z, u.z, -f.z, 0),
        SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)
    )
}

/// Inverse-transpose of the upper-left 3x3, embedded back into a 4x4 for cheap
/// passing to the shader. Only the 3x3 part is meaningful.
func normalMatrix(of model: simd_float4x4) -> simd_float4x4 {
    let m3 = simd_float3x3(
        SIMD3(model.columns.0.x, model.columns.0.y, model.columns.0.z),
        SIMD3(model.columns.1.x, model.columns.1.y, model.columns.1.z),
        SIMD3(model.columns.2.x, model.columns.2.y, model.columns.2.z)
    )
    let n3 = m3.inverse.transpose
    return simd_float4x4(
        SIMD4(n3.columns.0, 0),
        SIMD4(n3.columns.1, 0),
        SIMD4(n3.columns.2, 0),
        SIMD4(0, 0, 0, 1)
    )
}
