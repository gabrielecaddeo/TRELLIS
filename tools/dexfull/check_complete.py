import json, os, sys
R="/projects/gcaddeo/inference/TRELLIS/dex-full"
g=json.load(open(f"{R}/view_groups.json"))["groups"]
need=lambda I:[f"{R}/renders_cond/{I}/000.png",f"{R}/renders_cond/{I}/000_mask1.png",f"{R}/renders_cond/{I}/000_mask2.png",
 f"{R}/data_pose_norm/{I}/{I}_f000_meta.json",f"{R}/data_pose_norm/{I}/sdfs/{I}_f000__object.npy",f"{R}/data_pose_norm/{I}/sdfs/{I}_f000__hand.npy",
 f"{R}/data_pose_norm/{I}/contacts/{I}_f000_contact_coords.npy",f"{R}/data_pose_norm/{I}/contacts/{I}_f000_dist_to_contact.npy"]
missing={}; n=0; bad_groups=set()
for k,mem in g.items():
    for m in mem:
        I=m["instance"]; n+=1
        for p in need(I):
            if not os.path.exists(p): missing.setdefault(os.path.basename(p).replace(I,"<I>"),[]).append(I); bad_groups.add(k)
        if n%50000==0: print(n, {a:len(b) for a,b in missing.items()}, flush=True)
out={"n_instances":n,"n_groups":len(g),"bad_groups":len(bad_groups),"missing_counts":{a:len(b) for a,b in missing.items()},"missing_examples":{a:b[:5] for a,b in missing.items()}}
json.dump(out,open(sys.argv[1],"w"),indent=1); print(json.dumps(out,indent=1))
