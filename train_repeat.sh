for i in `seq 1 10`
do
    poetry run python myrl/train_dqn.py configs/ddqn.json
    poetry run python myrl/train_dqn.py configs/dqn.json
done
