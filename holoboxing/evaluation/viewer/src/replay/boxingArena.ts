// @ts-nocheck
import { REPLAY_ASSETS_BASE } from './assetPaths'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'

// Use the current evaluation arena and reject recordings from removed scenes.
export async function createBoxingArena(
  version?: string
): Promise<{ group: any; dispose(): void }> {
  if (version !== 'smplolympics-boxing-ring-v2') {
    throw new Error('当前评测播放器仅支持 smplolympics-boxing-ring-v2 场景')
  }
  const ringBase = `${REPLAY_ASSETS_BASE}g1-boxing/smpl-ring-v2/`
  const ringResponse = await fetch(`${ringBase}scene.json`)
  if (!ringResponse.ok) {
    throw new Error('SMPLOlympics 拳台资源加载失败')
  }
  const ringSource = await ringResponse.json()
  const [ringGroup, floorTexture] = await Promise.all([
    new OBJLoader().loadAsync(`${ringBase}${ringSource.mesh_file}`),
    new THREE.TextureLoader().loadAsync(
      `${REPLAY_ASSETS_BASE}g1-boxing/dark-arena/floor-v1.webp`
    )
  ])
  floorTexture.colorSpace = THREE.SRGBColorSpace
  floorTexture.anisotropy = 2
  const metal = new THREE.MeshStandardMaterial({
    color: '#354257',
    roughness: 0.65,
    metalness: 0.35
  })
  const pad = new THREE.MeshStandardMaterial({
    color: '#253244',
    roughness: 0.88,
    metalness: 0.03
  })
  const floor = new THREE.MeshStandardMaterial({
    map: floorTexture,
    roughness: 0.95,
    metalness: 0
  })
  // Use the recorded arena HSV S=0 rendering theme. Preserve V,
  // logos, geometry and all robot colors; do not desaturate the whole canvas.
  for (const material of [metal,pad,floor]) {
    material.onBeforeCompile = (shader: any) => {
      shader.fragmentShader=shader.fragmentShader.replace('#include <dithering_fragment>',
        '#include <dithering_fragment>\nfloat arenaValue=max(gl_FragColor.r,max(gl_FragColor.g,gl_FragColor.b));\ngl_FragColor.rgb=vec3(arenaValue);')
    }
    material.customProgramCacheKey=()=> 'original-arena-hsv-s-zero-v1'
  }
  ringGroup.userData.background='#2a2a2a'
  ringGroup.position.fromArray(ringSource.visual.position)
  ringGroup.rotation.set(...ringSource.visual.rotation)
  ringGroup.scale.fromArray(ringSource.visual.scale)
  ringGroup.updateMatrixWorld(true)
  ringGroup.userData.theme = 'holomotion-hsv-zero-v1'
  ringGroup.userData.floorTextureSize = 1024
  ringGroup.traverse((object: any) => {
    if (!object.isMesh) {
      return
    }
    const previous = Array.isArray(object.material)
      ? object.material
      : [object.material]
    previous.forEach((material: any) => material.dispose())
    if (object.name === '3DXY_geometry_001') {
      const positions = object.geometry.getAttribute('position')
      const uv = new Float32Array(positions.count * 2)
      const point = new THREE.Vector3()
      for (let i = 0; i < positions.count; i++) {
        point
          .fromBufferAttribute(positions, i)
          .applyMatrix4(object.matrixWorld)
        uv[2 * i] = (point.x + 3.05) / 6.1
        uv[2 * i + 1] = (point.y + 3.05) / 6.1
      }
      object.geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
      object.material = floor
    } else {
      object.material = /^3DXY_geometry_00[3-6]$/.test(object.name)
        ? pad
        : metal
    }
    object.castShadow = true
    object.receiveShadow = true
  })
  return {
    group: ringGroup,
    dispose() {
      ringGroup.traverse((object: any) => object.geometry?.dispose())
      floorTexture.dispose()
      floor.dispose()
      metal.dispose()
      pad.dispose()
    }
  }
}
