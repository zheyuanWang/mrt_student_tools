trap "exit" INT TERM ERR
trap "kill 0" EXIT

sequences=('00' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10')


for seq in "${sequences[@]}"
do
(   #tmp output folder on server:
    outputpath=/tmp/gridmap_comb12_${seq}/
    # move the output to local_path:
    local_path="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_baseline_comb12/${seq}/"


    fusion_template="${HOME}/6thsceneshot/src/sceneshot_tool/res/pointcloud_mapper/kitti/modifyPara/parameters_comb12_fusion.yaml"
    fusion_path="/tmp/params_fusion_${seq}.yaml"
    cp $fusion_template $fusion_path
    sed -i "s/{seq}/${seq}/g" $fusion_path
    ${HOME}/6thsceneshot/devel/lib/sceneshot_tool/pointcloud_mapper --parameters $fusion_path --output $outputpath


    intensity_template="${HOME}/6thsceneshot/src/sceneshot_tool/res/pointcloud_mapper/kitti/modifyPara/parameters_comb12_intensity.yaml"
    intensity_path="/tmp/params_intensity_${seq}.yaml"
    cp $intensity_template $intensity_path
    sed -i "s/{seq}/${seq}/g" $intensity_path
    ${HOME}/6thsceneshot/devel/lib/sceneshot_tool/pointcloud_mapper --parameters $intensity_path --output $outputpath


    mkdir -p ${local_path}
#    temp_path="/tmp/grid_map_${seq}"
#    mv $outputpath $temp_path
    rsync -r --progress $outputpath ${local_path}

#    bash ${HOME}/code/semantic-kitti-api/bash_create_gridmaps.sh "${seq}"
) &
done

wait
#DEUBG
#may need to delete the old folders in /tmp/ on server. if "Directory not empty"
