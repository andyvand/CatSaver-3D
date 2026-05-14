// CAT3D Metal 4 shaders. This file is compiled at runtime via
// MTL4Compiler.makeLibrary(descriptor:) — no offline metallib step required.

#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// Starfield background — a fullscreen triangle painted before the cat. Stars
// are procedural: per-cell hash decides whether the cell has a star, where
// inside the cell it sits, how bright it is, and how it twinkles. Three
// parallax layers scroll at different speeds.
// --------------------------------------------------------------------------

struct StarVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex StarVertexOut vertex_starfield(uint vid [[vertex_id]])
{
    // Fullscreen triangle: three corners that cover [-1,1] x [-1,1].
    float2 p = float2(float((vid << 1u) & 2u), float(vid & 2u));
    StarVertexOut out;
    out.position = float4(p * 2.0f - 1.0f, 1.0f, 1.0f);
    out.uv = p; // 0..1 inside the visible quad; >1 is clipped.
    return out;
}

static inline float starHash(float2 p)
{
    p = fract(p * float2(123.34f, 456.21f));
    p += dot(p, p + 45.32f);
    return fract(p.x * p.y);
}

// Mirrors the Swift-side ShaderTypes.h structs byte-for-byte.

struct CatVertex {
    float4 position;        // xyz used
    float4 normal;          // xyz used
    float4 uv;              // xy used
};

struct FrameUniforms {
    float4x4 viewProjection;
    float4   cameraPosTime;     // xyz=camera pos, w=time
    float4   hueParams;         // x=hueSpeed
    float4   keyLightDir;
    float4   keyLightColor;
    float4   fillLightDir;
    float4   fillLightColor;
    float4   ambientColor;
};

struct PartUniforms {
    float4x4 model;
    float4x4 normalMatrix;
    float4   baseColor;
    float4   furParams;         // x=shellCount, y=furLength, z=hueOffset, w=tint
    float4   miscParams;        // x=doFur
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 worldNormal;
    float2 uv;
    float  shellT;
    float  hueOffset;
    float  doFur;
    float  tint;
    float3 baseColor;
};

vertex VertexOut vertex_main(uint vid                       [[vertex_id]],
                              uint iid                       [[instance_id]],
                              device const CatVertex *verts  [[buffer(0)]],
                              constant FrameUniforms &frame  [[buffer(1)]],
                              constant PartUniforms  &part   [[buffer(2)]])
{
    CatVertex v = verts[vid];

    float shellCount = max(part.furParams.x, 1.0f);
    float furLength  = part.furParams.y;
    float doFur      = part.miscParams.x;

    float t = (shellCount > 1.0f) ? (float(iid) / (shellCount - 1.0f)) : 0.0f;

    float3 vpos = v.position.xyz;
    float3 vnor = v.normal.xyz;
    float2 vuv  = v.uv.xy;

    float3 pos = vpos + vnor * furLength * t;

    // Subtle wind-style sway, only on fur shells.
    float swayX = sin(frame.cameraPosTime.w * 1.3f + vpos.y * 5.0f) * 0.010f;
    float swayZ = cos(frame.cameraPosTime.w * 1.1f + vpos.x * 5.0f) * 0.010f;
    pos.x += swayX * t * doFur;
    pos.z += swayZ * t * doFur;

    float4 worldPos = part.model * float4(pos, 1.0f);
    float4 clipPos  = frame.viewProjection * worldPos;

    float3x3 nMtx = float3x3(part.normalMatrix[0].xyz,
                             part.normalMatrix[1].xyz,
                             part.normalMatrix[2].xyz);
    float3 worldNormal = normalize(nMtx * vnor);

    VertexOut out;
    out.position    = clipPos;
    out.worldPos    = worldPos.xyz;
    out.worldNormal = worldNormal;
    out.uv          = vuv;
    out.shellT      = t;
    out.hueOffset   = part.furParams.z;
    out.doFur       = doFur;
    out.tint        = part.furParams.w;
    out.baseColor   = part.baseColor.xyz;
    return out;
}

static inline float hash21(float2 p)
{
    p = fract(p * float2(123.34f, 456.21f));
    p += dot(p, p + 45.32f);
    return fract(p.x * p.y);
}

static inline float3 hsv2rgb(float3 c)
{
    float4 K = float4(1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0f);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), c.y);
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                               constant FrameUniforms &frame [[buffer(1)]])
{
    float t = in.shellT;

    // Per-strand mask: hash UV into cells, give each a random max strand length.
    float2 cell = floor(in.uv * 240.0f);
    float strandLen = mix(0.42f, 1.0f, hash21(cell));

    if (in.doFur > 0.5f && t > 0.001f && t > strandLen) {
        discard_fragment();
    }

    // Smooth rainbow rotation through the warm/green/teal half of the wheel,
    // skipping blues/purples/magentas entirely (hues ~0.60–1.00 in HSV).
    // We use a triangle wave so the cycle bounces 0 → 0.55 → 0 → ... and
    // never crosses into purple even at the wrap point.
    float phase = fract(frame.cameraPosTime.w * (1.0f / 44.0f) + in.hueOffset);
    float tri   = 1.0f - abs(phase * 2.0f - 1.0f);   // 0..1..0
    float hue   = tri * 0.55f;                       // 0..0.55..0 (red→teal)
    float3 rainbow = hsv2rgb(float3(hue, 0.70f, 0.95f));

    float shade  = (in.doFur > 0.5f) ? mix(0.55f, 1.05f, t) : 1.0f;
    float jitter = (in.doFur > 0.5f) ? mix(0.88f, 1.12f, hash21(cell + 13.0f)) : 1.0f;

    float3 surface = mix(in.baseColor, rainbow, in.tint) * shade * jitter;

    // Blinn-Phong with one key light + one fill + ambient.
    float3 N = normalize(in.worldNormal);
    float3 V = normalize(frame.cameraPosTime.xyz - in.worldPos);
    float3 L1 = normalize(frame.keyLightDir.xyz);
    float3 L2 = normalize(frame.fillLightDir.xyz);

    float diff1 = max(dot(N, L1), 0.0f);
    float diff2 = max(dot(N, L2), 0.0f);

    float3 ambient  = surface * frame.ambientColor.xyz;
    float3 diffuse  = surface * (diff1 * frame.keyLightColor.xyz +
                                 diff2 * frame.fillLightColor.xyz);

    float3 H = normalize(L1 + V);
    float spec = pow(max(dot(N, H), 0.0f), 18.0f) * 0.18f;
    float3 specular = float3(spec) * frame.keyLightColor.xyz;

    return float4(ambient + diffuse + specular, 1.0f);
}

fragment float4 fragment_starfield(StarVertexOut in [[stage_in]],
                                    constant FrameUniforms &frame [[buffer(1)]])
{
    float t  = frame.cameraPosTime.w;
    float2 uv = in.uv;

    // Subtle deep-space gradient — top navy, bottom slightly darker teal.
    // Avoiding any blue/red dominance so the background doesn't pulse purple.
    float3 col = mix(float3(0.005f, 0.015f, 0.030f),
                     float3(0.010f, 0.025f, 0.040f),
                     uv.y);

    // Drifting nebula clouds (very faint) in teal-cyan to stay clear of the
    // cat's warm rainbow range.
    {
        float2 n = uv * 3.0f + float2(t * 0.010f, t * 0.006f);
        float cloud = 0.5f + 0.5f * sin(n.x * 1.7f + sin(n.y * 2.3f) * 1.5f);
        cloud *= 0.5f + 0.5f * sin(n.y * 2.1f - n.x * 0.9f);
        col += float3(0.010f, 0.040f, 0.050f) * cloud;
    }

    // Stars across four parallax layers, each scrolling at a different speed
    // and direction. Denser + brighter than the first pass so they're
    // clearly visible against the background.
    const int LAYERS = 4;
    for (int layer = 0; layer < LAYERS; ++layer) {
        float lf       = float(layer);
        float speed    = 0.020f + 0.045f * lf;
        float density  = 80.0f  + 60.0f  * lf;
        float strength = 1.0f - 0.18f * lf;

        // Bias x by ~1.6 so cells stay roughly square on widescreen aspects.
        float2 p = float2(uv.x * density * 1.6f, uv.y * density);
        // Each layer drifts in a slightly different direction.
        float ang = 0.25f + lf * 0.45f;
        p += t * speed * density * float2(cos(ang), sin(ang) * 0.4f);

        float2 cell = floor(p);
        float2 f    = fract(p);

        float h = starHash(cell + 11.0f * lf);
        // ~12% of cells contain a star (was ~3.5%).
        if (h > 0.88f) {
            float2 starPos = float2(starHash(cell + 17.0f),
                                    starHash(cell + 43.0f));
            float d = length(f - starPos);

            float h2     = starHash(cell + 91.0f);
            float radius = mix(0.040f, 0.110f, h2);
            float core   = smoothstep(radius, 0.0f, d);
            float halo   = smoothstep(radius * 5.0f, radius, d) * 0.35f;

            // Diffraction "spikes" for the brightest stars — a thin cross.
            float spike = 0.0f;
            if (h2 > 0.7f) {
                float2 d2 = (f - starPos);
                float sx = smoothstep(0.005f, 0.0f, abs(d2.y)) *
                           smoothstep(radius * 6.0f, 0.0f, abs(d2.x));
                float sy = smoothstep(0.005f, 0.0f, abs(d2.x)) *
                           smoothstep(radius * 6.0f, 0.0f, abs(d2.y));
                spike = (sx + sy) * 0.35f * (h2 - 0.7f) / 0.3f;
            }

            float twinkle = 0.45f + 0.55f * sin(t * 2.5f + h * 30.0f);
            float3 tint   = mix(float3(0.80f, 0.90f, 1.00f),  // blue-white
                                float3(1.00f, 0.85f, 0.65f),  // amber
                                h2);
            float brightness = 1.6f * strength * twinkle;
            col += tint * (core * brightness + halo * brightness * 0.6f + spike);
        }
    }

    return float4(col, 1.0f);
}
