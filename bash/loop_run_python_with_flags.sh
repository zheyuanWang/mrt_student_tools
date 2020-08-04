set -e
SRC_DICT="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange10/"
#sequences=('00' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10')
sequences=('07')

for SQ in "${sequences[@]}"
do

echo "Processing SQ ${SQ}"
python ./semanticVoxelPNG2Grid.py --input "${SRC_DICT}/${SQ}/fusion/maps" --colorized 1 --no_dyn_objs 1
#python ./semanticVoxelPNG2Grid.py --input "${SRC_DICT}/${SQ}/single_shot/polar" --colorized 1

done