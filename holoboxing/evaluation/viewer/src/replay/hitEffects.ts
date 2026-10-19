import * as THREE from 'three'

// Reference 18202: attacker-colored core, eight sparks, expanding ring,
// local target glow and floating +1. Replay time also makes MP4 reproducible.
export function createHitEffects(scene: any,camera: any,objects: any[],canvas: HTMLCanvasElement) {
  const colors=['#ff405b','#3aa7ff']
  const effects=new Map<number,{group:any,event:any}>()
  const knockouts=new Map<number,any>()
  canvas.dataset.hitEffectColors=JSON.stringify(colors)
  const side=(value:any)=>value==='red'||value===0?0:value==='blue'||value===1?1:-1
  const disposeGroup=(group:any)=>{
    scene.remove(group)
    group.traverse((child:any)=>{child.geometry?.dispose();child.material?.map?.dispose();child.material?.dispose()})
  }
  const clear=()=>{
    effects.forEach(e=>disposeGroup(e.group));effects.clear()
    knockouts.forEach(disposeGroup);knockouts.clear()
    canvas.dataset.activeHitEffects='0';canvas.dataset.activeKoEffects='0'
  }
  const create=(event:any)=>{
    const actor=side(event.actor),color=colors[actor],group=new THREE.Group()
    group.position.fromArray(event.position)
    const material=()=>new THREE.MeshBasicMaterial({color,transparent:true,depthWrite:false,depthTest:false})
    group.add(new THREE.Mesh(new THREE.SphereGeometry(.045,12,8),material()))
    const ring=new THREE.Mesh(new THREE.TorusGeometry(.08,.009,6,32),material())
    ring.userData.billboard=true;group.add(ring)
    for(let i=0;i<8;i++){
      const spark=new THREE.Mesh(new THREE.SphereGeometry(.009,6,4),material())
      spark.position.set(Math.cos(i*Math.PI/4)*.06,Math.sin(i*Math.PI/4)*.06,(i%3-1)*.03)
      group.add(spark)
    }
    const label=document.createElement('canvas');label.width=384;label.height=96
    const ctx=label.getContext('2d')!
    ctx.font='bold 64px sans-serif';ctx.textAlign='center';ctx.fillStyle=color
    ctx.shadowColor='#0b1525';ctx.shadowBlur=10;ctx.fillText('击中 +1',192,68)
    const text=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(label),transparent:true,depthTest:false,depthWrite:false}))
    text.scale.set(.48,.12,1);text.userData.scoreLabel=true;group.add(text)
    group.children.forEach((c:any)=>c.renderOrder=1000)
    scene.add(group);return group
  }
  const createKnockout=(actor:number)=>{
    const label=document.createElement('canvas');label.width=640;label.height=320
    const ctx=label.getContext('2d')!
    ctx.textAlign='left';ctx.strokeStyle='#101725'
    ctx.fillStyle=actor<0?'#f1f5f9':colors[actor]
    ctx.shadowColor='#000000';ctx.shadowBlur=10
    // Fit visible glyph bounds, including CJK font fallback, to identical edges.
    const line=(text:string,baseline:number)=>{
      ctx.font='900 160px sans-serif'
      let bounds=ctx.measureText(text)
      const width=bounds.actualBoundingBoxLeft+bounds.actualBoundingBoxRight
      ctx.font=`900 ${160*400/width}px sans-serif`
      bounds=ctx.measureText(text)
      ctx.lineWidth=7
      const x=120+bounds.actualBoundingBoxLeft
      ctx.strokeText(text,x,baseline);ctx.fillText(text,x,baseline)
      return [x-bounds.actualBoundingBoxLeft,x+bounds.actualBoundingBoxRight]
    }
    const titleBounds=line(actor<0?'DRAW':'K.O.',162)
    const subtitleBounds=line(actor<0?'本场平局':(actor===0?'红方':'蓝方')+'获胜',282)
    canvas.dataset.resultTextBounds=JSON.stringify({title:titleBounds,subtitle:subtitleBounds})
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(label),transparent:true,depthTest:false,depthWrite:false}))
    sprite.renderOrder=2000;scene.add(sprite);return sprite
  }
  const render=(events:any[],frame:number,time:number,times:number[],result?:{winner?:string})=>{
    const wanted=new Set<number>();let count=0,last:any
    for(const object of objects){object.mesh.material.emissive?.set('#000000');object.mesh.material.emissiveIntensity=0}
    events.forEach((event,index)=>{
      if(event.type!=='HIT'||event.frame>frame||side(event.actor)<0)return
      count++;last=event
      if(!Array.isArray(event.position)||event.position.length!==3||!event.position.every(Number.isFinite))return
      const elapsed=time-times[event.frame]
      if(elapsed<0||elapsed>=1.4)return
      wanted.add(index)
      if(!effects.has(index))effects.set(index,{group:create(event),event})
      const group=effects.get(index)!.group,age=elapsed/1.4,color=colors[side(event.actor)]
      group.scale.setScalar(1+age*2.2)
      group.children.forEach((child:any)=>{
        child.material.opacity=1-age
        if(child.userData.scoreLabel)child.position.z=.12+age*.13
        if(child.userData.billboard)child.quaternion.copy(camera.quaternion)
      })
      let nearest:any,distance=Infinity
      for(const object of objects)if(object.side===side(event.target)){
        const d=object.mesh.position.distanceTo(group.position)
        if(d<distance){distance=d;nearest=object.mesh}
      }
      if(nearest&&distance<.5){nearest.material.emissive.set(color);nearest.material.emissiveIntensity=(1-age)*1.5}
    })
    for(const [id,effect] of effects)if(!wanted.has(id)){disposeGroup(effect.group);effects.delete(id)}
    const koActors=new Set<number>()
    for(const event of events){
      if(event.type!=='KO'||event.frame>frame||side(event.actor)<0)continue
      const elapsed=time-times[event.frame]
      if(elapsed>=0&&elapsed<2)koActors.add(side(event.actor))
    }
    const terminal=frame===times.length-1
    const draw=(koActors.size>1 || (terminal&&result?.winner==='draw'))
    const shown=new Set<number>(draw?[-1]:[...koActors].slice(0,1))
    camera.updateMatrixWorld()
    for(const actor of shown){
      if(!knockouts.has(actor))knockouts.set(actor,createKnockout(actor))
      const sprite=knockouts.get(actor)
      sprite.position.set(0,0,.5).unproject(camera)
      const depth=sprite.position.clone().applyMatrix4(camera.matrixWorldInverse).z
      const height=2*Math.tan(camera.fov*Math.PI/360)*Math.abs(depth)
      const width=Math.min(height*camera.aspect*.86,height*.82*2)
      sprite.scale.set(width,width/2,1)
    }
    for(const [actor,sprite] of knockouts)if(!shown.has(actor)){disposeGroup(sprite);knockouts.delete(actor)}
    canvas.dataset.activeKoEffects=String(shown.size)
    canvas.dataset.koEffectActors=JSON.stringify([...shown])
    canvas.dataset.koEffectColors=JSON.stringify([...shown].map(actor=>actor<0?'#f1f5f9':colors[actor]))
    canvas.dataset.resultEffect=draw?'DRAW':shown.size?'K.O.':''
    canvas.dataset.activeHitEffects=String(effects.size)
    canvas.dataset.hitEventCount=String(count)
    if(last){canvas.dataset.lastHitEffectActor=String(side(last.actor));canvas.dataset.lastHitEffectColor=colors[side(last.actor)]}
  }
  return {render,dispose:clear}
}
