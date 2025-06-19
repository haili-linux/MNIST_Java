
import java.util.ArrayList;

import haili.deeplearn.utils.MatrixUtil;
import haili.deeplearn.utils.ThreadWork;


public class Test {

    static QLearning2 qLearning2 = new QLearning2("qLearnMode.txt");

    static ArrayList<float[]> statesList = new ArrayList<>();
    static ArrayList<float[]> actList = new ArrayList<>();
    static ArrayList<Float> rewardList = new ArrayList<>();
    static ArrayList<Integer> gameOverList = new ArrayList<>();
    static boolean test = false;
    static float Beta = 0.9f;
    static int multi_step = 2;
    static int trainDataLen =  1024 * 64;
    static int dataLen = trainDataLen * multi_step + multi_step;


    public static void main(String[] args) throws Exception{

        System.out.println(qLearning2.model_main.loss);
        SnakeGame snakeGame = new SnakeGame();
        snakeGame.initGame();


        long time = System.currentTimeMillis();
        int n = 0;
        int step = 0;
        while (true) {
            snakeGame.VIEW = test;
            snakeGame.render();

            float[] state = snakeGame.state();
            float[] act = qLearning2.sample(state,  test);
            statesList.add(state);
            actList.add(act);

            snakeGame.doAction(act);
            snakeGame.update();

            rewardList.add(snakeGame.reward());
            gameOverList.add(snakeGame.gameOver ? 0:1);

            if (!test && multi_step > 1) {
                for (int i = 1; i < multi_step; i++) {
                    if (snakeGame.gameOver) {
                        statesList.add(new float[0]);
                        actList.add(new float[0]);
                        rewardList.add(0f);
                        gameOverList.add(0);

                    } else {
                        snakeGame.render();
                        float[] state_ti = snakeGame.state();
                        float[] act_ti = qLearning2.sample(state, true);
                        snakeGame.doAction(act_ti);
                        snakeGame.update();
                        statesList.add(state_ti);
                        actList.add(act_ti);
                        rewardList.add(snakeGame.reward());
                        gameOverList.add(snakeGame.gameOver ? 0:1);
                    }
                }
            }

            removeData();

            if(snakeGame.gameOver){
                n++;
                snakeGame.initGame();
            }







            if (!test) {
                long nowtime = System.currentTimeMillis();
                if (nowtime - time > 10 * 60 * 1000) {
                    qLearning2.save("qLearnMode.txt");
                    time = nowtime;
                }

                step ++;
                if (statesList.size() == dataLen && step >= 1024 * 8) {
                    qLearning2.updateModel_Target();
                    step = 0;
                    training();
                }

            } else {
                Thread.sleep(200);
            }
        }

    }

    public static void removeData() {
        if (statesList.size() > dataLen) {
            for (int i = 0; i < multi_step; i++) {
                statesList.remove(0);
                actList.remove(0);
                rewardList.remove(0);
                gameOverList.remove(0);
            }
        }
    }

    public static void training() {
        if (rewardList.isEmpty())
            return;

        float[][] train_x = new float[trainDataLen][];
        float[][] train_y = new float[trainDataLen][];

//        for (int i = 0; i < train_x.length; i++) {
//            train_x[i] = MatrixUtil.combine(statusList.get(i), actList.get(i));
//
//            if (gameOverList.get(i) == 1 && i < train_x.length - 1)
//                train_y[i] = new float[]{ sorceList.get(i) + 0.9f * qLearning2.getMaxSorce(statusList.get(i + 1))};
//            else
//                train_y[i] = new float[]{ sorceList.get(i) };
//        }

        ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(trainDataLen) {
            @Override
            public void working(int index) {
                int index_base = index * multi_step;
                train_x[index] = MatrixUtil.combine(statesList.get(index_base), actList.get(index_base));
                if (train_x[index].length < 1) {
                    System.out.println();
                }

//                if (gameOverList.get(index) == 1 && index < train_x.length - 1)
//                    train_y[index] = new float[]{ rewardList.get(index) + Beta * qLearning2.getMaxSorce(statesList.get(index + 1))};
//                else
//                    train_y[index] = new float[]{ rewardList.get(index) };

                train_y[index] = new float[1];
                float beta = 1f;
                for (int i = 0; i < multi_step; i++) {
                    int index_d = index_base + i;
                    if (gameOverList.get(index_d) == 1) {
                        train_y[index][0] += beta * rewardList.get(index_d);
                    } else {
                        train_y[index][0] += beta * rewardList.get(index_d);
                        return;
                    }
                    beta *= Beta;
                }

                train_y[index][0] += beta * qLearning2.getMaxSorce(statesList.get( index_base + multi_step ));
            }

        };
        ThreadWork.start(threadWorker, 24);

        System.out.println(qLearning2.model_main.calculateLoss(train_x, train_y));
        qLearning2.model_main.fit(train_x, train_y, 128, 5, 24);
    }

}