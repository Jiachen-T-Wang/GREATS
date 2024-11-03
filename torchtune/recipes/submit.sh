


Method=( "Regular" "MaxLoss" "GradNorm" "TracIN-AdaptiveSelect-PerBatch") 

Batch=(16 8 )


for method in "${Method[@]}"; do
    for bs in "${Batch[@]}"; do
        echo "method: $method"
        echo "batch size: $bs"
        sbatch run_batch.sh $method $bs
    done
done





