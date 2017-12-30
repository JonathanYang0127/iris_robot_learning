#!/bin/bash
# tmp.csv should be a CSV with two columns: instance id and public DNS (IP)

while IFS=, read -r col1 col2
do
    instanceid=$col1
    ip=$col2
    scp -i /home/vitchyr/git/doodad/aws_config/private/key_pairs/doodad-us-west-1.pem \
        -oStrictHostKeyChecking=no \
        ubuntu@$ip:/tmp/doodad-output/variant.json /tmp/variant.json\
        > /dev/null
    # This kills all instance where the version is equal to "DDPG-TDM"
    value=$(cat /tmp/variant.json | jq '.algo_kwargs.num_updates_per_env_step')
    echo $value
    if [ $value -eq 20 ]; then
        aws ec2 terminate-instances --instance-ids $instanceid
    else
        echo OKAY
    fi
done < $1