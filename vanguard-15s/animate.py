"""Continuous 15-second cinematic treatment of the new Vanguard headset render.

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

OUT = Path('/workspace/vanguard_headset_15s')
SOURCE = Path('/workspace/generated_images/exec-bd23f036-ee63-46a5-946f-0357f41c57ee.png')
W, H, FPS, SECONDS = 1152, 768, 24, 15
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

cup=polygon([(900,806),(912,735),(940,648),(966,567),(999,503),(1031,468),(1071,458),(1121,467),(1170,491),(1213,522),(1240,556),(1244,600),(1231,669),(1211,739),(1188,807),(1154,856),(1106,884),(1047,901),(985,905),(939,890),(910,857)])
left_band=polygon([(541,283),(552,246),(575,201),(601,164),(638,123),(662,104),(650,125),(615,164),(589,201),(567,247),(555,289)])
right_band=polygon([(674,88),(719,60),(769,40),(821,30),(872,29),(923,38),(979,58),(1034,87),(1080,127),(1121,171),(1154,217),(1183,271),(1204,326),(1218,375),(1219,401),(1183,397),(1149,384),(1118,374),(1098,319),(1075,261),(1044,209),(1008,161),(966,121),(918,89),(868,65),(818,49),(770,49),(722,65)])
far_cup=polygon([(506,379),(473,400),(447,433),(432,476),(420,524),(416,583),(422,638),(438,697),(463,746),(494,774),(533,790),(509,757),(489,713),(478,660),(474,602),(478,538),(488,482),(508,424),(532,379)])
mask=np.maximum.reduce([cup,left_band,right_band,far_cup])
# A warm ivory highlight should never become animated camouflage. The broad
# polygons also keep controls and joints outside the animated surface.
sat=(base.max(2)-base.min(2))/(base.max(2)+1e-5)
colour_gate=np.clip((sat-.12)/.28,0,1)
mask*=colour_gate
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
spill=np.clip((red-green-.06)/.24,0,1)*np.clip((yy/H-.84)/.10,0,1)
spill*=np.clip((.9-np.abs(xx/W-.5))/.4,0,1)
spill=gaussian_filter(spill,5)

mp4=OUT/'vanguard-headset-15-seconds.mp4'
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
    # Continuous, non-repeating flow over the complete fifteen-second shot.
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
    frame+=glow[:,:,None]*np.array([.72,.32,1.0],dtype=np.float32)
    # A light traveling through the material has a small corresponding soft halo.
    halo=gaussian_filter(np.maximum(glow,0),4.5)
    frame+=halo[:,:,None]*np.array([.22,.045,.13],dtype=np.float32)
    reflection=(float(np.sum(light*mask)/(mask.sum()+1))-.5)*.12
    frame+=spill[:,:,None]*reflection*np.array([1.,.35,.65],dtype=np.float32)
    image=Image.fromarray(np.uint8(np.clip(frame,0,1)*255))
    # A continuous camera push and lateral drift; no pretend hidden 3D angles.
    ease=u*u*(3-2*u)
    zoom=.91+.08*ease
    roll=math.radians(.32*math.cos(math.pi*u))
    ca,sa=math.cos(roll)/zoom,math.sin(roll)/zoom
    cx=W/2+6*math.sin(math.pi*u)
    cy=H/2-3*math.sin(math.pi*u)
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
    if n in [0,120,240,359]:result.save(OUT/f'review-{n:03d}.png')
    if n%60==0: print(f'Frame {n+1}/360 | {time.monotonic()-started:.1f}s',flush=True)
encoder.stdin.close()
if encoder.wait()!=0: raise SystemExit('Video encoding failed')

gif=OUT/'vanguard-headset-15-seconds.gif'
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(mp4),
 '-filter_complex','fps=12,scale=512:-2:flags=lanczos,split[a][b];'
 '[a]palettegen=max_colors=192:stats_mode=full[p];'
 '[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
 '-loop','0',str(gif)],check=True)
report={'source':str(SOURCE),'type':'Animated product rendering with camera motion and flowing shell illumination',
        'duration_seconds':15,'continuous_shot':True,'files':[]}
for p,count in [(mp4,360),(gif,180)]:
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
      '-count_frames','-show_entries','stream=width,height,nb_read_frames:format=duration',
      '-of','json',str(p)],text=True))
    assert int(info['streams'][0]['nb_read_frames'])==count,info
    assert abs(float(info['format']['duration'])-15)<.025,info
    report['files'].append({'path':str(p),'bytes':p.stat().st_size,**info})
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
