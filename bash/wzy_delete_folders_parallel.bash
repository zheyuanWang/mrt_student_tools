#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

#use the bash in the dataset root folder, where 00, 01 ... sequences are 
CURRENT_DIR=$(pwd)
MID_NAME="r8"

sequences=('00' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10')
#sequences=('06' '07' '08' '09' '10')
for seq in "${sequences[@]}"

do
(	
	echo "moving ${seq} to /tmp/${MID_NAME} on server"
	mkdir -p /tmp/"${MID_NAME}"/"${seq}"
	mv "${CURRENT_DIR}"/"${seq}"/fusion/maps/ground_surface/ /tmp/"${MID_NAME}"/"${seq}"/
	mv "${CURRENT_DIR}"/"${seq}"/fusion/maps/ground_distance_z_max/ /tmp/"${MID_NAME}"/"${seq}"/
	mv "${CURRENT_DIR}"/"${seq}"/fusion/maps/ground_distance_z_min/ /tmp/"${MID_NAME}"/"${seq}"/
	mv "${CURRENT_DIR}"/"${seq}"/fusion/maps/object_boundaries/ /tmp/"${MID_NAME}"/"${seq}"/
	mv "${CURRENT_DIR}"/"${seq}"/fusion/maps/bba_free/ /tmp/"${MID_NAME}"/"${seq}"/
	mv "${CURRENT_DIR}"/"${seq}"/fusion/maps/bba_occupied/ /tmp/"${MID_NAME}"/"${seq}"/
)&
done


wait

