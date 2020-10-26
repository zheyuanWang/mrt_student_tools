trap "exit" INT TERM ERR
trap "kill 0" EXIT

vor_path="/mrtstorage/users/zwang/pcd_mapper_pastonly"
folder_list=('polar_single' 'polar_pastrange5+1' 'polar_pastrange8' 'polar_pastrange12' 'polar_pastrange15' 'polar_pastrange20')
seq='00'
nach_path="/${seq}/fusion/maps/learning_semantic_grid_dense_colorized"

image='000879.png'

for folder in "${folder_list[@]}"
do
(
path="${vor_path}/${folder}/${nach_path}"
echo "${folder}"

cp ${path}/${image} /tmp/${folder}_${seq}_${image}

) &
done

wait
