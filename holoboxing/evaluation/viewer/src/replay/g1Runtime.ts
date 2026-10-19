// @ts-nocheck
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { loadG1Assets } from './g1Assets'
import { createBoxingArena } from './boxingArena'
import { ReplayData } from './Hdf5ReplayReader'
import { createHitEffects } from './hitEffects'

export type G1Camera = 'broadcast' | 'ringside' | 'overhead'
export interface G1Runtime {
  canvas: HTMLCanvasElement
  render(frame: number): void
  resize(width: number, height: number): void
  setCamera(camera: G1Camera): void
  diagnostics(): {
    qposRed: number[]
    qposBlue: number[]
    meshCount: number
    frame: number
    jointAddresses: Record<string, number>
    rootAddress: number
  }
  dispose(): void
}

export async function createG1Runtime(
  replay: ReplayData,
  canvas?: HTMLCanvasElement,
  signal?: AbortSignal,
  onStage?: (label: string) => void
): Promise<G1Runtime> {
  const checkAbort = () => {
    if (signal?.aborted) {
      throw new DOMException('回放已切换', 'AbortError')
    }
  }
  checkAbort()
  const arenaVersion = replay.meta.environment?.asset_version
  if (arenaVersion !== 'smplolympics-boxing-ring-v2') {
    throw new Error('当前评测播放器仅支持 smplolympics-boxing-ring-v2 场景')
  }
  const started = performance.now()
  const timing: Record<string, number> = {}
  let previous = started
  const mark = (name: string) => {
    const now = performance.now()
    timing[name] = Math.round(now - previous)
    previous = now
  }
  onStage?.('正在下载机器人资源…')
  const variant = 'g1-boxing'
  const { mj, manifest } = await loadG1Assets(variant)
  mark('assetsAndEngineMs')
  onStage?.('正在准备机器人模型…')
  await new Promise(resolve => setTimeout(resolve, 0))
  checkAbort()
  const models: any[] = []
  try {
    for (const side of ['red', 'blue']) {
      const model = mj.MjModel.mj_loadXML(
        `/${variant}-assets/${manifest.entries?.[side] || manifest.entry}`
      )
      if (!model) {
        throw new Error('MuJoCo 无法编译 G1 MJCF')
      }
      model.opt.disableflags |=
        mj.mjtDisableBit.mjDSBL_CONTACT.value |
        mj.mjtDisableBit.mjDSBL_CONSTRAINT.value
      models.push(model)
    }
  } catch (error) {
    models.forEach(model => model.delete())
    throw error
  }
  mark('modelCompileMs')
  onStage?.('正在准备场地和首帧…')
  const robotData: any[] = []
  const geometries = new Map<string, any>()
  const scene = new THREE.Scene()
  let renderer: any,
    controls: any,
    arena: Awaited<ReturnType<typeof createBoxingArena>> | undefined
  let currentFrame = 0,
    disposed = false
  let hitEffects: ReturnType<typeof createHitEffects> | undefined
  const dispose = () => {
    if (disposed) {
      return
    }
    disposed = true
    hitEffects?.dispose()
    controls?.dispose()
    const releasedGeometry = new Set<any>(),
      releasedMaterial = new Set<any>()
    scene.traverse((object: any) => {
      if (object.geometry && !releasedGeometry.has(object.geometry)) {
        object.geometry.dispose()
        releasedGeometry.add(object.geometry)
      }
      for (const material of object.material
        ? Array.isArray(object.material)
          ? object.material
          : [object.material]
        : []) {
        if (!releasedMaterial.has(material)) {
          material.dispose()
          releasedMaterial.add(material)
        }
      }
    })
    arena?.dispose()
    renderer?.dispose()
    renderer?.forceContextLoss()
    robotData.forEach(data => data.delete())
    models.forEach(model => model.delete())
  }
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    })
    renderer.setPixelRatio(1)
    renderer.setSize(1280, 720, false)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.15
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    scene.background = new THREE.Color('#f0f4fa')
    scene.fog = new THREE.Fog('#f0f4fa', 14, 30)
    const camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.05, 80)
    camera.up.set(0, 0, 1)
    camera.position.set(2.4, -3.0, 2.1)
    const target = new THREE.Vector3(0, 0, 0.85)
    camera.lookAt(target)
    if (canvas) {
      controls = new OrbitControls(camera, canvas)
      controls.target.copy(target)
      controls.enableDamping = false
      controls.minDistance = 2
      controls.maxDistance = 14
      controls.maxPolarAngle = Math.PI * 0.47
      controls.addEventListener('change', () => {
        if (!disposed) {
          renderer.render(scene, camera)
        }
      })
    }
    const hemi = new THREE.HemisphereLight(0xf0f5ff, 0x303645, 2.7)
    hemi.position.set(0, 0, 8)
    scene.add(hemi)
    const key = new THREE.DirectionalLight(0xfff8ec, 3.5)
    key.position.set(3, -4, 7)
    key.castShadow = true
    key.shadow.mapSize.set(2048, 2048)
    key.shadow.camera.left = -4
    key.shadow.camera.right = 4
    key.shadow.camera.top = 4
    key.shadow.camera.bottom = -4
    key.shadow.normalBias = 0.025
    scene.add(key)
    const rim = new THREE.DirectionalLight(0x9dc3ff, 2)
    rim.position.set(-3, 4, 4)
    scene.add(rim)

    arena = await createBoxingArena(arenaVersion)
    renderer.domElement.dataset.arenaVersion = arenaVersion
    scene.add(arena.group)
    if (arena.group.userData.theme) {
      scene.background = new THREE.Color(arena.group.userData.background || '#2a2a2a')
      scene.fog = new THREE.Fog(arena.group.userData.background || '#2a2a2a', 14, 30)
      renderer.domElement.dataset.arenaTheme = arena.group.userData.theme
      renderer.domElement.dataset.floorTextureSize = String(
        arena.group.userData.floorTextureSize
      )
    }
    renderer.domElement.dataset.arenaObjects = JSON.stringify(
      arena.group.children.map((object: any) => object.name)
    )
    checkAbort()
    const model = models[0]
    const freeJoint = Array.from(model.jnt_type as ArrayLike<number>).findIndex(
      type => type === 0
    )
    if (freeJoint < 0) {
      throw new Error('G1 MJCF 缺少 free joint')
    }
    const rootAddress = model.jnt_qposadr[freeJoint]
    const objects: Array<{ mesh: any; geom: number; side: number }> = []
    const jointAddresses: Record<string, number> = {}
    const mappings: Array<{
      root: number
      quat: number
      joints: number
      addresses: number[]
    }> = []
    for (const [sideIndex, side] of ['red', 'blue'].entries()) {
      const model = models[sideIndex]
      const schema = replay.meta.state_schema.robots?.[side]
      const names: string[] =
        schema?.joint_names || replay.meta.state_schema.joint_names
      const fields = schema?.fields
      if (
        !fields ||
        !Array.isArray(names) ||
        names.length !== 29 ||
        new Set(names).size !== 29
      ) {
        throw new Error('G1 回放缺少29关节名称及状态布局')
      }
      for (const [field, size] of [
        ['root_position', 3],
        ['root_quaternion', 4],
        ['joint_position', 29]
      ] as const) {
        const offset = fields[field]?.offset
        if (
          !Number.isInteger(offset) ||
          offset < 0 ||
          offset + size > replay.stateWidth
        ) {
          throw new Error(`G1 回放状态布局无效：${field}`)
        }
      }
      const addresses = names.map(name => {
        const joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, name)
        if (joint < 0) {
          throw new Error(`G1 MJCF 缺少回放关节 ${name}`)
        }
        jointAddresses[name] = model.jnt_qposadr[joint]
        return model.jnt_qposadr[joint]
      })
      mappings.push({
        root: fields.root_position.offset,
        quat: fields.root_quaternion.offset,
        joints: fields.joint_position.offset,
        addresses
      })
      robotData.push(new mj.MjData(model))
      for (let geom = 0; geom < model.ngeom; geom++) {
        // Official mesh geoms only; collision capsules/spheres are not robot appearance.
        if (
          model.geom_type[geom] !== mj.mjtGeom.mjGEOM_MESH.value ||
          model.geom_group[geom] !== 1
        ) {
          continue
        }
        const meshId = model.geom_dataid[geom]
        const geometryKey = `${sideIndex}:${meshId}`
        let geometry = geometries.get(geometryKey)
        if (!geometry) {
          const firstVertex = model.mesh_vertadr[meshId] * 3,
            vertexCount = model.mesh_vertnum[meshId] * 3
          const firstFace = model.mesh_faceadr[meshId] * 3,
            faceCount = model.mesh_facenum[meshId] * 3
          geometry = new THREE.BufferGeometry()
          geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(
              new Float32Array(
                model.mesh_vert.slice(firstVertex, firstVertex + vertexCount)
              ),
              3
            )
          )
          geometry.setIndex(
            new THREE.BufferAttribute(
              new Uint32Array(
                model.mesh_face.slice(firstFace, firstFace + faceCount)
              ),
              1
            )
          )
          geometry.computeVertexNormals()
          geometries.set(geometryKey, geometry)
        }
        const matId = model.geom_matid[geom]
        const rgba =
          matId >= 0
            ? model.mat_rgba.slice(matId * 4, matId * 4 + 4)
            : model.geom_rgba.slice(geom * 4, geom * 4 + 4)
        const original = new THREE.Color(rgba[0], rgba[1], rgba[2])
        const color = original.clone()
        if (variant === 'g1' && (rgba[0] + rgba[1] + rgba[2]) / 3 > 0.35) {
          color.lerp(
            new THREE.Color(sideIndex === 0 ? '#e65265' : '#448dee'),
            0.7
          )
        }
        const material = new THREE.MeshStandardMaterial({
          color,
          roughness: 0.42,
          metalness: 0.28
        })
        const mesh = new THREE.Mesh(geometry, material)
        mesh.castShadow = true
        mesh.receiveShadow = true
        scene.add(mesh)
        objects.push({ mesh, geom, side: sideIndex })
      }
    }
    if (objects.length < 40) {
      throw new Error('G1 真实网格加载不完整')
    }
    const matrix = new THREE.Matrix4()
    hitEffects=createHitEffects(scene,camera,objects,renderer.domElement)
    const render = (frame: number) => {
      if (disposed) {
        return
      }
      currentFrame = Math.max(0, Math.min(replay.time.length - 1, frame))
      const base = currentFrame * replay.stateWidth
      robotData.forEach((data, side) => {
        const mapping = mappings[side]
        for (let i = 0; i < 3; i++) {
          data.qpos[rootAddress + i] = replay.state[base + mapping.root + i]
        }
        for (let i = 0; i < 4; i++) {
          data.qpos[rootAddress + 3 + i] = replay.state[base + mapping.quat + i]
        }
        mapping.addresses.forEach((address, j) => {
          data.qpos[address] = replay.state[base + mapping.joints + j]
        })
        data.time = replay.time[currentFrame]
        mj.mj_forward(models[side], data)
      })
      for (const object of objects) {
        const data = robotData[object.side],
          p = object.geom * 3,
          r = object.geom * 9
        object.mesh.position.set(
          data.geom_xpos[p],
          data.geom_xpos[p + 1],
          data.geom_xpos[p + 2]
        )
        matrix.set(
          data.geom_xmat[r],
          data.geom_xmat[r + 1],
          data.geom_xmat[r + 2],
          0,
          data.geom_xmat[r + 3],
          data.geom_xmat[r + 4],
          data.geom_xmat[r + 5],
          0,
          data.geom_xmat[r + 6],
          data.geom_xmat[r + 7],
          data.geom_xmat[r + 8],
          0,
          0,
          0,
          0,
          1
        )
        object.mesh.quaternion.setFromRotationMatrix(matrix)
      }
      hitEffects.render(replay.events,currentFrame,replay.time[currentFrame],replay.time,replay.meta.result)
      renderer.render(scene, camera)
    }
    const setCamera = (preset: G1Camera) => {
      camera.position.set(
        ...(preset === 'overhead'
          ? [0.01, -0.01, 8]
          : preset === 'ringside'
          ? [0.1, -4.8, 1.7]
          : [2.4, -3.0, 2.1])
      )
      camera.lookAt(target)
      controls?.target.copy(target)
      controls?.update()
      render(currentFrame)
    }
    const resize = (width: number, height: number) => {
      if (disposed || width < 1 || height < 1) {
        return
      }
      renderer.setSize(Math.round(width), Math.round(height), false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.domElement.dataset.projectionAspect = String(camera.aspect)
      render(currentFrame)
    }
    mark('sceneBuildMs')
    render(0)
    mark('firstRenderMs')
    timing.totalMs = Math.round(performance.now() - started)
    renderer.domElement.dataset.loadTimings = JSON.stringify(timing)
    console.info('[G1 replay load]', timing)
    return {
      canvas: renderer.domElement,
      render,
      resize,
      setCamera,
      diagnostics: () => ({
        frame: currentFrame,
        qposRed: Array.from(robotData[0].qpos),
        qposBlue: Array.from(robotData[1].qpos),
        meshCount: objects.length,
        jointAddresses,
        rootAddress
      }),
      dispose
    }
  } catch (error) {
    dispose()
    throw error
  }
}
