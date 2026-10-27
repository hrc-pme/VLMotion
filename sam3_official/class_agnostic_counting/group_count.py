#!/usr/bin/env python3
"""Unknown-category instance grouping: generic SAM3 proposals + CLIP clustering."""

from __future__ import annotations

import argparse
import colorsys
import gc
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, Sam3Model, Sam3Processor


SAM3 = "/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
CLIP = "/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--output", type=Path, default=Path("/workspace/Outputs/class_agnostic_counting"))
    p.add_argument("--prompts", nargs="+", default=["surgical instrument", "metal surgical tool"])
    p.add_argument("--score", type=float, default=0.18)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--nms-iou", type=float, default=0.72)
    p.add_argument("--cluster-distance", type=float, default=0.18)
    p.add_argument("--diagnostic-roi", nargs=4, type=float, metavar=("X1","Y1","X2","Y2"))
    p.add_argument("--diagnostic-pair", nargs=2, type=int, metavar=("ID_A","ID_B"))
    p.add_argument("--diagnostic-only", action="store_true")
    p.add_argument("--diagnostic-raw-ids", nargs="+", type=int)
    return p.parse_args()


def iomask(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_iom(a, b):
    smaller = min(int(a.sum()), int(b.sum()))
    return float(np.logical_and(a, b).sum() / smaller) if smaller else 0.0


def split_parent_evidence(proposals):
    """Find a union-like parent that hides two distinct child instances."""
    split_parents = {}
    for parent_index, parent in enumerate(proposals):
        px1,py1,px2,py2=parent["box"]
        pw,ph=max(px2-px1,1),max(py2-py1,1)
        children=[]
        for child_index,child in enumerate(proposals):
            if child_index==parent_index or child["prompt"]!=parent["prompt"]:
                continue
            area_ratio=child["area"]/max(parent["area"],1)
            if not .25<=area_ratio<=.75 or mask_iom(child["mask"],parent["mask"])<.82:
                continue
            children.append((child_index,child))
        best=None
        for i in range(len(children)):
            for j in range(i+1,len(children)):
                ai,a=children[i]; bi,b=children[j]
                if iomask(a["mask"],b["mask"])>.28:
                    continue
                ac=np.array([(a["box"][0]+a["box"][2])/2,(a["box"][1]+a["box"][3])/2])
                bc=np.array([(b["box"][0]+b["box"][2])/2,(b["box"][1]+b["box"][3])/2])
                separation=np.linalg.norm((ac-bc)/np.array([pw,ph]))
                union=np.logical_or(a["mask"],b["mask"])
                coverage=np.logical_and(union,parent["mask"]).sum()/max(parent["area"],1)
                if separation>=.18 and coverage>=.70:
                    quality=coverage+separation+.1*(a["score"]+b["score"])
                    if best is None or quality>best[0]: best=(quality,ai,bi,coverage,separation)
        if best:
            split_parents[parent_index]={"children":[best[1],best[2]],
                                         "coverage":float(best[3]),"separation":float(best[4])}
    return split_parents


def propose(image, cfg, device):
    processor = Sam3Processor.from_pretrained(SAM3, local_files_only=True)
    model = Sam3Model.from_pretrained(SAM3, local_files_only=True).to(device=device, dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    proposals = []
    image_area = image.width * image.height
    for prompt in cfg.prompts:
        print(f"SAM3: {prompt}", flush=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            out = model(**inputs)
        result = processor.post_process_instance_segmentation(
            out, threshold=cfg.score, mask_threshold=cfg.mask_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        for score, box, mask in zip(result["scores"], result["boxes"], result["masks"]):
            mask = mask.cpu().numpy().astype(bool)
            area = int(mask.sum())
            if image_area * 0.00015 <= area <= image_area * 0.12:
                proposals.append({
                    "score": float(score.cpu()), "box": box.float().cpu().tolist(),
                    "mask": mask, "area": area, "prompt": prompt,
                })
    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    if cfg.diagnostic_roi:
        rx1,ry1,rx2,ry2=cfg.diagnostic_roi
        print("Raw candidates centered in diagnostic ROI:", flush=True)
        for index,item in enumerate(sorted(proposals,key=lambda x:x["score"],reverse=True),1):
            x1,y1,x2,y2=item["box"]; cx,cy=(x1+x2)/2,(y1+y2)/2
            if rx1<=cx<=rx2 and ry1<=cy<=ry2:
                print({"raw_id":index,"score":round(item["score"],4),
                       "box":[round(v,1) for v in item["box"]],"area":item["area"],
                       "prompt":item["prompt"]}, flush=True)
    if cfg.diagnostic_raw_ids:
        ordered_items=sorted(proposals,key=lambda x:x["score"],reverse=True)
        canvas=image.convert("RGBA")
        overlay=Image.new("RGBA",image.size,(0,0,0,0)); draw=ImageDraw.Draw(overlay)
        colors=[(255,60,60),(50,220,120),(60,140,255),(255,190,30),(190,70,240),(0,220,220)]
        for color_index,raw_id in enumerate(cfg.diagnostic_raw_ids):
            if not 1<=raw_id<=len(ordered_items): continue
            item=ordered_items[raw_id-1]; color=colors[color_index%len(colors)]
            alpha=Image.fromarray((item["mask"]*95).astype(np.uint8))
            layer=Image.new("RGBA",image.size,color+(0,)); layer.putalpha(alpha); overlay.alpha_composite(layer)
            x1,y1,x2,y2=item["box"]
            draw.rectangle((x1,y1,x2,y2),outline=color+(255,),width=4)
            draw.text((x1+3,max(0,y1-24)),f"raw {raw_id} {item['score']:.3f}",fill=color+(255,),stroke_width=2,stroke_fill=(0,0,0,255))
        diagnostic_dir=cfg.output/cfg.image.stem/"diagnostics"; diagnostic_dir.mkdir(parents=True,exist_ok=True)
        diagnostic=Image.alpha_composite(canvas,overlay).convert("RGB"); diagnostic.thumbnail((1800,1800))
        diagnostic.save(diagnostic_dir/"selected_raw_candidates.jpg",quality=95)
    split_parents=split_parent_evidence(proposals)
    rejected_parent_ids=set(split_parents)
    for parent_index,evidence in split_parents.items():
        parent=proposals[parent_index]
        children=[proposals[i] for i in evidence["children"]]
        print("Split-parent evidence:", {"parent_score":round(parent["score"],4),
              "parent_box":[round(v,1) for v in parent["box"]],
              "child_scores":[round(x["score"],4) for x in children],
              "child_boxes":[[round(v,1) for v in x["box"]] for x in children],
              "coverage":round(evidence["coverage"],3),
              "separation":round(evidence["separation"],3)}, flush=True)
    kept = []
    ordered=sorted(enumerate(proposals), key=lambda x: x[1]["score"], reverse=True)
    for original_index,item in ordered:
        if original_index in rejected_parent_ids:
            continue
        if not any(
            iomask(item["mask"], old["mask"]) >= cfg.nms_iou
            or mask_iom(item["mask"], old["mask"]) >= 0.88
            for old in kept
        ):
            kept.append(item)
    print(f"Candidates: raw={len(proposals)}, split_parents={len(split_parents)}, after_nms={len(kept)}", flush=True)
    for item in kept:
        item["split_parent_signal"]=False
    for evidence in split_parents.values():
        for child_index in evidence["children"]:
            child=proposals[child_index]
            for item in kept:
                if iomask(item["mask"],child["mask"])>=.72:
                    item["split_parent_signal"]=True
    return kept


def masked_crop(rgb, item):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = item["box"]
    padx, pady = (x2-x1)*.06, (y2-y1)*.06
    xa, ya, xb, yb = max(0,int(x1-padx)), max(0,int(y1-pady)), min(w,int(x2+padx)), min(h,int(y2+pady))
    crop = rgb[ya:yb, xa:xb].copy()
    crop[~item["mask"][ya:yb, xa:xb]] = 127
    return Image.fromarray(crop)


def shape(item, size):
    x1,y1,x2,y2 = item["box"]; bw=max(x2-x1,1); bh=max(y2-y1,1)
    contours,_ = cv2.findContours(item["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea) if contours else None
    hull=cv2.contourArea(cv2.convexHull(contour)) if contour is not None else item["area"]
    per=cv2.arcLength(contour,True) if contour is not None else 0
    return [math.log(bw/bh), math.log(item["area"]/(size[0]*size[1])), item["area"]/max(hull,1),
            item["area"]/max(bw*bh,1), min(per*per/max(4*math.pi*item["area"],1),30)/30]


def cluster(image, items, cfg, device):
    if len(items) < 2: return np.zeros(len(items), dtype=int)
    proc = CLIPImageProcessor.from_pretrained(CLIP, local_files_only=True)
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP, local_files_only=True).to(device)
    model.eval().requires_grad_(False)
    crops=[masked_crop(np.asarray(image), x) for x in items]
    vec=[]
    for i in range(0,len(crops),8):
        pixels=proc(images=crops[i:i+8], return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            batch=torch.nn.functional.normalize(model(pixel_values=pixels).image_embeds, dim=-1)
        vec.append(batch.cpu().numpy())
    visual=np.concatenate(vec)
    shaped=StandardScaler().fit_transform(np.asarray([shape(x,image.size) for x in items]))
    centers=np.asarray([[(x["box"][0]+x["box"][2])/(2*image.width),(x["box"][1]+x["box"][3])/(2*image.height)] for x in items])
    visual_distance=1-visual@visual.T
    shape_distance=np.linalg.norm(shaped[:,None,:]-shaped[None,:,:],axis=-1)/math.sqrt(shaped.shape[1])
    spatial_distance=np.linalg.norm(centers[:,None,:]-centers[None,:,:],axis=-1)
    # Similar metal appearance alone must not merge spatially separate trays or
    # regions. Shape and position are supporting evidence, with a hard spatial
    # gate that prevents the previous large chain-like cluster.
    distances=(.80*visual_distance + .10*np.clip(shape_distance/3,0,1)
               + .10*np.clip(spatial_distance/.20,0,1))
    distances[spatial_distance>.22]=1.0
    np.fill_diagonal(distances,0)
    if cfg.diagnostic_pair:
        a,b=(cfg.diagnostic_pair[0]-1,cfg.diagnostic_pair[1]-1)
        if 0<=a<len(items) and 0<=b<len(items):
            print("Pair diagnostic:", {"ids":[a+1,b+1],
                  "visual_distance":round(float(visual_distance[a,b]),4),
                  "shape_distance":round(float(shape_distance[a,b]),4),
                  "spatial_distance":round(float(spatial_distance[a,b]),4),
                  "combined_distance":round(float(distances[a,b]),4),
                  "cluster_threshold":cfg.cluster_distance}, flush=True)
    upper=distances[np.triu_indices(len(items),1)]
    print("Pairwise cosine-distance quantiles:",
          {str(q):round(float(np.quantile(upper,q)),4) for q in (0,.1,.25,.5,.75,.9,1)}, flush=True)
    return AgglomerativeClustering(n_clusters=None,metric="precomputed",linkage="complete",
                                   distance_threshold=cfg.cluster_distance).fit_predict(distances)


def group_name(n):
    return f"Group {chr(65+n)}" if n < 26 else f"Group {n+1}"


def unique_group_color(index):
    """Return one deterministic RGB color per visual group."""
    hue = (0.07 + index * 0.618033988749895) % 1.0
    saturation = (0.78, 0.92, 0.66)[index % 3]
    value = (0.96, 0.82)[(index // 3) % 2]
    return tuple(
        int(round(channel * 255))
        for channel in colorsys.hsv_to_rgb(hue, saturation, value)
    )


def overlay_font(image_width):
    size = max(16, image_width // 70)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save(image, items, labels, cfg):
    run=cfg.output/cfg.image.stem; run.mkdir(parents=True,exist_ok=True)
    canvas=image.convert("RGBA"); overlay=Image.new("RGBA",image.size,(0,0,0,0)); draw=ImageDraw.Draw(overlay)
    remap={old:i for i,old in enumerate(sorted(set(map(int,labels))))}
    label_counts={label:int(sum(remap[int(raw)]==label for raw in labels)) for label in remap.values()}
    font=overlay_font(image.width)
    font_size=getattr(font,"size",max(16,image.width//70))
    instances=[]
    for i,(item,raw) in enumerate(zip(items,labels),1):
        label=remap[int(raw)]; color=unique_group_color(label)
        alpha=Image.fromarray((item["mask"]*85).astype(np.uint8))
        layer=Image.new("RGBA",image.size,color+(0,)); layer.putalpha(alpha); overlay.alpha_composite(layer)
        x1,y1,x2,y2=item["box"]; draw.rectangle((x1,y1,x2,y2),outline=color+(255,),width=max(3,image.width//1200))
        draw.text(
            (x1+4,max(0,y1-font_size-8)),
            f"{group_name(label)} n={label_counts[label]} | ID {i} | {item['score']:.2f}",
            font=font, fill=color+(255,), stroke_width=2, stroke_fill=(0,0,0,255),
        )
        instances.append({"id":i,"group":group_name(label),"score":item["score"],"box_xyxy":item["box"],"prompt":item["prompt"],
                          "group_color_rgb":list(color),
                          "intervention_signal":"possible_merged_instances" if item.get("split_parent_signal") else None})
    counts={group_name(label):label_counts[label] for label in sorted(label_counts)}
    summary=[f"Total count: {len(items)}"] + [f"{name}: {count}" for name,count in counts.items()]
    line_height=font_size+7
    summary_width=max(draw.textbbox((0,0),line,font=font)[2] for line in summary)+20
    summary_height=line_height*len(summary)+12
    draw.rounded_rectangle((8,8,8+summary_width,8+summary_height),radius=8,fill=(0,0,0,190))
    for line_index,line in enumerate(summary):
        draw.text((18,14+line_index*line_height),line,font=font,fill=(255,255,255,255))
    out=Image.alpha_composite(canvas,overlay).convert("RGB"); out.thumbnail((1800,1800)); out.save(run/"grouped_overlay.jpg",quality=95)
    result={"mode":"unknown_category_visual_grouping_prototype","total":len(items),"group_counts":counts,"instances":instances,
            "warning":"Anonymous visual groups are not medical instrument classifications."}
    (run/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"overlay":str(run/"grouped_overlay.jpg"),**result},ensure_ascii=False,indent=2))


def main():
    cfg=args(); device=torch.device("cuda")
    image=Image.open(cfg.image).convert("RGB")
    if max(image.size) > 2048:
        scale=2048/max(image.size)
        image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
    items=propose(image,cfg,device)
    if cfg.diagnostic_only:
        return
    labels=cluster(image,items,cfg,device); save(image,items,labels,cfg)


if __name__ == "__main__": main()
