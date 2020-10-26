set -e
sequences=('00' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10')
#sequences=('06' '07' '08' '09' '10' )
#sequences=( '00' '01' '02' '03' '04' '05' )


for seq in "${sequences[@]}"
do

    local_path="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange8/${seq}/fusion/maps/semantic_full/"

    mkdir -p ${local_path}
    temp_path="/tmp/grid_map_${seq}/fusion/maps/semantic_full/"
    rsync -r --progress $temp_path ${local_path}

done

#DEUBG
#may need to delete the old folders in /tmp/ on server. if "Directory not empty"
