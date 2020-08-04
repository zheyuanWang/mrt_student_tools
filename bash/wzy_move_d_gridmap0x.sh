set e
# move all the content of girdmap to its root folder
WORKDIR=$(pwd)
sequences=('00' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10')
#sequences=('06' '07' '08' '09' '10')
for seq in "${sequences[@]}"
do
	#cd ./seq/grid_map_${seq}/
	mv ${WORKDIR}/${seq}/grid_map_${seq}/* ${WORKDIR}/${seq}/
done
