#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// Argument-table buffer slots — shared between MSL and the Swift renderer.
typedef enum CatBufferIndex {
    CatBufferIndexVertices = 0,
    CatBufferIndexFrame    = 1,
    CatBufferIndexPart     = 2,
} CatBufferIndex;

typedef struct {
    simd_float4 position;   // xyz used (w = 1)
    simd_float4 normal;     // xyz used (w = 0)
    simd_float4 uv;         // xy used  (zw = 0)
} CatVertex;

typedef struct {
    simd_float4x4 viewProjection;
    simd_float4   cameraPosTime;   // xyz = camera position, w = time
    simd_float4   hueParams;       // x = hueSpeed
    simd_float4   keyLightDir;     // xyz used
    simd_float4   keyLightColor;   // xyz used
    simd_float4   fillLightDir;    // xyz used
    simd_float4   fillLightColor;  // xyz used
    simd_float4   ambientColor;    // xyz used
} FrameUniforms;

typedef struct {
    simd_float4x4 model;
    simd_float4x4 normalMatrix;    // 4x4 holding a 3x3 (upper-left) for convenience
    simd_float4   baseColor;       // xyz used
    simd_float4   furParams;       // x=shellCount, y=furLength, z=hueOffset, w=tint
    simd_float4   miscParams;      // x=doFur
} PartUniforms;

#endif
