set -e
sequences=('00' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10')
#sequences=( '10' '09' )
#sequences=('06' '07' '08' '09' '10' )
#sequences=( '00' '01' '02' '03' '04' '05' )
#sequences=( '05' '06' )
#sequences=( '03' '02' )
#sequences=( '05' '06' ) # knecht4
#sequences=( '00' )

for seq in "${sequences[@]}"
do
    #move the output to local_path:
    local_path="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange12_move/${seq}/"

    mkdir -p ${local_path}
    temp_path="/tmp/grid_map_${seq}"
#   mv /tmp/grid_map/ $temp_path
    rsync -r --progress $temp_path ${local_path}

#    bash ${HOME}/code/semantic-kitti-api/bash_create_gridmaps.sh "${seq}"
done

#DEUBG
#may need to delete the old folders in /tmp/ on server. if "Directory not empty"
