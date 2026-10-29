"""Continuous 30-second cinematic treatment of the new Verdant headset render.

Animated material and light are composited inside the optical shells. The
stationary source artwork is not evidence of a physically working product.
"""
from pathlib import Path
import json
import math
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter, distance_transform_edt

OUT = Path('/workspace/verdant_headset_30s')
SOURCE = Path('/workspace/generated_images/exec-e45cdd94-2b9c-468f-a76a-438fd2f54f5c.png')
W, H, FPS, SECONDS = 1152, 768, 24, 30
OUT.mkdir(exist_ok=True)
original = Image.open(SOURCE).convert('RGB')
original.save(OUT / 'design.png')
source = original.resize((W,H),Image.Resampling.LANCZOS)
base = np.asarray(source,dtype=np.float32)/255
sx, sy = W/original.width, H/original.height

def polygon(points):
    canvas=Image.new('L',(W,H),0)
    ImageDraw.Draw(canvas).polygon([(round(x*sx),round(y*sy)) for x,y in points],fill=255)
    return np.asarray(canvas,dtype=np.float32)/255

cup=polygon([(416,516),(425,490),(450,482),(500,481),(548,490),(585,511),(609,546),(630,593),(645,651),(655,711),(657,759),(646,793),(625,811),(591,819),(552,816),(514,806),(482,788),(461,763),(446,726),(433,674),(425,623),(416,565)])
left_band=polygon([(455,401),(467,352),(490,303),(521,260),(557,226),(599,200),(645,180),(695,168),(746,165),(796,167),(849,179),(891,196),(929,223),(961,256),(988,294),(1009,337),(1026,381),(1030,402),(1018,406),(1005,361),(982,314),(955,276),(925,244),(888,218),(848,197),(801,181),(754,174),(711,180),(670,196),(630,220),(593,251),(563,292),(539,342),(520,401)])
right_band=polygon([(641,258),(686,244),(733,238),(778,242),(823,255),(863,278),(899,313),(920,352),(937,395),(948,407),(940,377),(924,339),(906,310),(880,283),(848,260),(811,245),(773,237),(731,234),(685,240)])
far_cup=polygon([(1040,481),(1062,502),(1080,532),(1091,562),(1095,604),(1090,652),(1078,702),(1058,748),(1037,775),(1009,792),(973,799),(1003,778),(1023,741),(1039,695),(1053,644),(1061,589),(1058,550),(1048,515)])
mask=np.maximum.reduce([cup,left_band,right_band,far_cup])
# A warm ivory highlight should never become animated camouflage. The broad
# polygons also keep controls and joints outside the animated surface.
sat=(base.max(2)-base.min(2))/(base.max(2)+1e-5)
colour_gate=np.clip((sat-.12)/.28,0,1)
green_gate=np.clip((base[:,:,1]-base[:,:,0]-.08)/.24,0,1)
mask*=colour_gate*green_gate
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
mask=gaussian_filter(mask,1.1)
mask=np.clip(mask,0,1)
Image.fromarray(np.uint8(mask*255)).save(OUT/'material-mask.png')
edge=np.clip(distance_transform_edt(mask>.15)/55,0,1).astype(np.float32)
# Optical highlights remain anchored to the shell surface rather than swimming.
neutral_highlight=np.clip((base.min(2)-.45)/.35,0,1)*np.clip((.35-sat)/.25,0,1)
flow_mask=mask*(1-.95*neutral_highlight)

regions=[]
for region in [cup,left_band,right_band,far_cup]:
    ys,xs=np.where(region>.5)
    x0,x1=max(0,xs.min()-16),min(W,xs.max()+17)
    y0,y1=max(0,ys.min()-16),min(H,ys.max()+17)
    regions.append((x0,x1,y0,y1))

# Restrict reflected-light animation to the existing colored pool on the table.
red,green,blue=base[:,:,0],base[:,:,1],base[:,:,2]
spill=np.clip((green-red-.06)/.24,0,1)*np.clip((yy/H-.84)/.10,0,1)
spill*=np.clip((.9-np.abs(xx/W-.5))/.4,0,1)
spill=gaussian_filter(spill,5)

mp4=OUT/'verdant-headset-30-seconds.mp4'
cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo',
     '-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),
     '-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','17',
     '-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)]
encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
started=time.monotonic()
for n in range(FPS*SECONDS):
    t=n/FPS
    u=t/(SECONDS-1/FPS)
    frame=base.copy()
    # Continuous, non-repeating flow over the complete thirty-second shot.
    for x0,x1,y0,y1 in regions:
        x=xx[y0:y1,x0:x1];y=yy[y0:y1,x0:x1]
        e=edge[y0:y1,x0:x1]
        dx=e*(21.0*np.sin(y*.014+x*.004-t*1.35)+7.0*np.sin(x*.022+t*.72))
        dy=e*(18.0*np.sin(x*.013-y*.005+t*1.12)+6.0*np.cos(y*.023-t*.91))
        coords=[y+dy,x+dx]
        warped=np.stack([map_coordinates(base[:,:,c],coords,order=1,mode='nearest')
                         for c in range(3)],axis=2)
        m=flow_mask[y0:y1,x0:x1,None]
        frame[y0:y1,x0:x1]=frame[y0:y1,x0:x1]*(1-m)+warped*m
    wave=.5+.5*np.sin(xx*.009+yy*.007-t*1.63+.35*np.sin(t*.53))
    secondary=.5+.5*np.cos(yy*.015-xx*.006+t*1.13)
    light=.62*wave+.38*secondary
    strength=(light-.47)*.30
    glow=mask*strength
    frame+=glow[:,:,None]*np.array([.18,1.0,.76],dtype=np.float32)
    # A light traveling through the material has a small corresponding soft halo.
    halo=gaussian_filter(np.maximum(glow,0),4.5)
    frame+=halo[:,:,None]*np.array([.025,.24,.15],dtype=np.float32)
    reflection=(float(np.sum(light*mask)/(mask.sum()+1))-.5)*.12
    frame+=spill[:,:,None]*reflection*np.array([.12,1.,.72],dtype=np.float32)
    image=Image.fromarray(np.uint8(np.clip(frame,0,1)*255))
    # A continuous camera push and lateral drift; no pretend hidden 3D angles.
    ease=u*u*(3-2*u)
    zoom=1.0+.13*ease
    roll=math.radians(.5*math.sin(2*math.pi*u))
    ca,sa=math.cos(roll)/zoom,math.sin(roll)/zoom
    cx=W/2+14*math.sin(2*math.pi*u)
    cy=H/2+8*math.sin(math.pi*u)
    pad=96
    padded_pixels=np.pad(np.asarray(image),((pad,pad),(pad,pad),(0,0)),mode='reflect')
    # Extend the seamless backdrop above the image; reflect only the empty
    # tabletop and sides, so the wider camera position has natural texture.
    padded_pixels[:pad]=padded_pixels[pad:pad+1]
    padded=Image.fromarray(padded_pixels)
    affine=(ca,-sa,cx+pad-ca*W/2+sa*H/2,
            sa,ca,cy+pad-sa*W/2-ca*H/2)
    result=padded.transform((W,H),Image.Transform.AFFINE,affine,
                            resample=Image.Resampling.BICUBIC)
    encoder.stdin.write(result.tobytes())
    if n in [0,240,480,719]:result.save(OUT/f'review-{n:03d}.png')
    if n%60==0: print(f'Frame {n+1}/720 | {time.monotonic()-started:.1f}s',flush=True)
encoder.stdin.close()
if encoder.wait()!=0: raise SystemExit('Video encoding failed')

gif=OUT/'verdant-headset-30-seconds.gif'
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(mp4),
 '-filter_complex','fps=10,scale=448:-2:flags=lanczos,split[a][b];'
 '[a]palettegen=max_colors=160:stats_mode=full[p];'
 '[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
 '-loop','0',str(gif)],check=True)
report={'source':str(SOURCE),'type':'Animated product rendering with camera motion and flowing shell illumination',
        'duration_seconds':30,'continuous_shot':True,'files':[]}
for p,count in [(mp4,720),(gif,300)]:
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
      '-count_frames','-show_entries','stream=width,height,nb_read_frames:format=duration',
      '-of','json',str(p)],text=True))
    assert int(info['streams'][0]['nb_read_frames'])==count,info
    assert abs(float(info['format']['duration'])-30)<.025,info
    report['files'].append({'path':str(p),'bytes':p.stat().st_size,**info})
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
