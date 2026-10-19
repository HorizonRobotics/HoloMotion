// @ts-nocheck
export interface ReplayData {
  displayParticipants?: { red: string; blue: string }
  meta: Record<string, any>
  time: number[]
  state: number[]
  stateWidth: number
  score: number[]
  round: number[]
  events: Array<{
    frame: number
    type: string
    actor: string
    target: string
    value: number
    [key: string]: any
  }>
  stateDataset: 'state' | 'mujoco_state'
}

let replaySequence = 0

const MAX_BYTES = 100 * 1024 * 1024
/** Copies data out of the WASM heap, closes HDF5 handles and removes temporary bytes. */
export async function readHdf5Replay(bytes: ArrayBuffer): Promise<ReplayData> {
  if (bytes.byteLength > MAX_BYTES) {
    throw new Error('开发播放器文件上限为 100 MiB')
  }
  const signature = new Uint8Array(bytes, 0, Math.min(8, bytes.byteLength))
  if (String(signature) !== '137,72,68,70,13,10,26,10') {
    throw new Error('文件不是有效的 HDF5 文件')
  }
  const { default: h5wasm } = await import('h5wasm')
  const { FS } = await h5wasm.ready
  const path = `/replay-${Date.now()}-${++replaySequence}.hdf5`
  FS.writeFile(path, new Uint8Array(bytes))
  let file: InstanceType<typeof h5wasm.File> | undefined
  try {
    file = new h5wasm.File(path, 'r')
    const dataset = (name: string) => {
      const value = file!.get(name) as any
      if (!(value instanceof h5wasm.Dataset)) {
        throw new Error(`缺少数据集 ${name}`)
      }
      // Bound decompressed allocation before materializing the dataset.
      const count = (value.shape || []).reduce(
        (n: number, v: number) => n * v,
        1
      )
      if (count > 10_000_000) {
        throw new Error(`数据集过大 ${name}`)
      }
      return value
    }
    const numeric = (name: string): number[] =>
      Array.from(dataset(name).value as ArrayLike<number | bigint>, Number)
    const strings = (name: string): string[] =>
      Array.from(dataset(name).value as ArrayLike<string>, String)
    const rawMeta = dataset('/meta/json').value
    const meta = JSON.parse(typeof rawMeta === 'string' ? rawMeta : rawMeta[0])
    if (!['1.0', '1.1'].includes(meta.schema_version)) {
      throw new Error(`不支持 replay schema ${meta.schema_version}`)
    }
    const time = numeric('/frames/time')
    const stateDataset = file.get('/frames/mujoco_state')
      ? 'mujoco_state'
      : 'state'
    const stateNode = dataset(`/frames/${stateDataset}`)
    const shape = stateNode.shape
    const state = numeric(`/frames/${stateDataset}`)
    if (meta.state_schema?.id === 'g1_kinematic_v1') {
      if (
        meta.state_schema.quaternion_order !== 'wxyz' ||
        meta.state_schema.coordinate_system !== 'right-handed-z-up'
      ) {
        throw new Error('G1 回放坐标约定不受支持')
      }
      for (const side of ['red', 'blue']) {
        const robot = meta.state_schema.robots?.[side]
        if (
          !Array.isArray(robot?.joint_names) ||
          robot.joint_names.length !== 29 ||
          new Set(robot.joint_names).size !== 29 ||
          robot.joint_names.some(name => typeof name !== 'string')
        ) {
          throw new Error('G1 回放需要29个唯一关节名称')
        }
        for (const [name, size] of [
          ['root_position', 3],
          ['root_quaternion', 4],
          ['joint_position', 29],
          ['joint_velocity', 29]
        ] as const) {
          const field = robot.fields?.[name]
          if (
            !field ||
            !Number.isInteger(field.offset) ||
            field.offset < 0 ||
            field.size !== size ||
            field.offset + size > shape?.[1]
          ) {
            throw new Error(`G1 状态字段无效：${side}.${name}`)
          }
        }
      }
    }
    const score = numeric('/frames/score')
    const round = numeric('/frames/round')
    if (
      !time.length ||
      shape?.length !== 2 ||
      shape[0] !== time.length ||
      !shape[1] ||
      score.length !== time.length * 2 ||
      round.length !== time.length
    ) {
      throw new Error('帧数据维度不一致')
    }
    if (
      ![time, state, score, round].every(values =>
        values.every(Number.isFinite)
      )
    ) {
      throw new Error('帧数据包含 NaN / Inf')
    }
    if (time.some((value, i) => i > 0 && value <= time[i - 1])) {
      throw new Error('时间轴必须严格递增')
    }
    if (
      meta.schema_version === '1.1' &&
      (meta.timebase?.mode !== 'explicit_timestamps' ||
        !Number.isFinite(meta.timebase.nominal_step_seconds) ||
        meta.timebase.nominal_step_seconds <= 0 ||
        time[0] !== 0)
    ) {
      throw new Error('回放时间基准无效')
    }
    if (meta.state_schema?.id === 'generic_articulated_replay_v1') {
      const entities = meta.state_schema.entities
      if (
        !entities ||
        typeof entities !== 'object' ||
        !Object.keys(entities).length
      ) {
        throw new Error('通用回放缺少实体布局')
      }
      for (const [name, entity] of Object.entries(entities) as Array<
        [string, any]
      >) {
        const count = entity.body_names?.length
        if (
          !count ||
          new Set(entity.body_names).size !== count ||
          entity.parent_indices?.length !== count
        ) {
          throw new Error(`实体拓扑无效：${name}`)
        }
        for (let i = 0; i < count; i++) {
          const seen = new Set<number>()
          let current = i
          while (current !== -1) {
            if (
              !Number.isInteger(current) ||
              current < 0 ||
              current >= count ||
              seen.has(current)
            ) {
              throw new Error(`实体父节点存在越界或环：${name}`)
            }
            seen.add(current)
            current = entity.parent_indices[current]
          }
        }
        for (const [fieldName, stride] of [
          ['body_world_position', 3],
          ['body_world_quaternion', 4]
        ] as const) {
          const field = entity.fields?.[fieldName]
          if (
            !field ||
            !Number.isInteger(field.offset) ||
            field.offset < 0 ||
            field.shape?.[0] !== count ||
            field.shape?.[1] !== stride ||
            field.offset + count * stride > shape[1]
          ) {
            throw new Error(`实体字段布局无效：${name}.${fieldName}`)
          }
        }
      }
    }

    const frames = numeric('/events/frame'),
      types = strings('/events/type'),
      actors = strings('/events/actor'),
      targets = strings('/events/target'),
      values = numeric('/events/value')
    if (
      ![types, actors, targets, values].every(v => v.length === frames.length)
    ) {
      throw new Error('事件数据维度不一致')
    }
    const details = file.get('/events/json') ? strings('/events/json').map(v=>JSON.parse(v)) : []
    if (details.length && details.length!==frames.length) throw new Error('事件详情数量不一致')
    const events = frames.map((frame, i) => {
      if (
        !Number.isInteger(frame) ||
        frame < 0 ||
        frame >= time.length ||
        !Number.isFinite(values[i])
      ) {
        throw new Error('事件帧索引或值无效')
      }
      return {
        ...details[i],
        frame,
        type: types[i],
        actor: actors[i],
        target: targets[i],
        value: values[i]
      }
    })
    return {
      meta,
      time,
      state,
      stateWidth: shape[1],
      score,
      round,
      events,
      stateDataset
    }
  } finally {
    file?.close()
    FS.unlink(path)
  }
}

export const Hdf5ReplayReader = { read: readHdf5Replay }
