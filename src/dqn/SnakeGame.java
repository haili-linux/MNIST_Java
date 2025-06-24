package dqn;

import java.awt.*;
import java.util.*;

import javax.swing.*;

public class SnakeGame {

    private static final char SNAKE_HEAD_CHAR = 'O';
    private static final char SNAKE_BODY_CHAR = 'o';
    private static final char FOOD_CHAR = '*';
    private static final char EMPTY_CHAR = ' ';
    private static final char WALL_CHAR = '#';
    static final char[] acts = {'W', 'S', 'A', 'D', ' '};

    int WIDTH;
    int HEIGHT;

    private final Deque<Point> snake;
    private Point food;
    private int eatFood = 0;
    private char direction = 'D';
    public boolean gameOver = false;
    private final Random random;
    public StringBuffer gameStr;

    public boolean VIEW = true;

    BoardDisplay boardDisplay;

    public SnakeGame(int width, int height){
        WIDTH = width;
        HEIGHT = height;
        snake = new LinkedList<>();
        random = new Random();
        gameStr = new StringBuffer();
        boardDisplay = new BoardDisplay();
        SwingUtilities.invokeLater(() -> {
            boardDisplay.setVisible(true);
        });
    }

    public float[] randomAction(){
        float[] act = new float[5];
        int index = (int)(act.length * Math.random());
        act[index] = 1f;
        return act;
    }


    public char floatToAct(float[] act){
        int max_index = 0;
        for(int i = 1; i < act.length; i++){
            if( act[i] > act[max_index])
                max_index = i;
        }

        return acts[max_index];
    }

    public float[] state(){
        String s = gameStr.toString();
        s = s.replace(WALL_CHAR,'\n').replace("\n", "");
        float[] r = new float[s.length() * 3];
        for(int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            switch (c) {
                case FOOD_CHAR:
                    r[i] = 1f;
                    break;
                case SNAKE_BODY_CHAR:
                    r[i + s.length() ] = 1f;
                    break;
                case SNAKE_HEAD_CHAR:
                    r[i + s.length() * 2] = 1f;
                    break;
                /*
                case WALL_CHAR:
                    r[i + s.length() * 3] = 1;
                    break;
                    */
            }
        }

        return r;
    }

    public void initGame() {
        gameOver = false;
        snake.clear();

        int x = WIDTH / 2;
        int y = HEIGHT / 2;

        snake.add(new Point(x, y));
        snake.add(new Point(x - 1, y));
        direction = 'D';
        generateFood();
    }

    private void generateFood() {
        while (true) {
            int x = random.nextInt(WIDTH);
            int y = random.nextInt(HEIGHT);
            Point p = new Point(x, y);
            if (!snake.contains(p)) {
                food = p;
                break;
            }
        }
    }


    public void update() {

        Point head = snake.peekFirst();

        if (head == null) return;

        Point newHead = new Point(head.x, head.y);

        switch (direction) {
            case 'W': newHead.y--; break;
            case 'S': newHead.y++; break;
            case 'A': newHead.x--; break;
            case 'D': newHead.x++; break;
        }

        if (newHead.x < 0 || newHead.y < 0 || newHead.x >= WIDTH || newHead.y >= HEIGHT || snake.contains(newHead)) {
            gameOver = true;
            return;
        }

        snake.addFirst(newHead);

        eatFood = 0;
        if (newHead.equals(food)) {
            generateFood();
            eatFood = 4;
        } else {
            snake.removeLast();
        }

    }

    public void render() {
        char[][] board = new char[HEIGHT][WIDTH];

        for (char[] row : board) Arrays.fill(row, EMPTY_CHAR);

        for (Point p : snake){
            if (p == snake.peekFirst())
                board[p.y][p.x] = SNAKE_HEAD_CHAR;
            else
                board[p.y][p.x] = SNAKE_BODY_CHAR;
        }

        board[food.y][food.x] = FOOD_CHAR;

        gameStr = new StringBuffer();

        for (int i = 0; i < WIDTH + 2; i++)
            gameStr.append(WALL_CHAR);

        gameStr.append('\n');

        for (int y = 0; y < HEIGHT; y++) {
            gameStr.append(WALL_CHAR);
            for (int x = 0; x < WIDTH; x++) {
                gameStr.append(board[y][x]);
            }
            gameStr.append(WALL_CHAR).append('\n');
        }

        for (int i = 0; i < WIDTH + 2; i++)
            gameStr.append(WALL_CHAR);

        if(VIEW){
            boardDisplay.setText(gameStr.toString());
        }
    }

    public void doAction(char input) {
        if ((input == 'W' && direction != 'S') ||
                (input == 'S' && direction != 'W') ||
                (input == 'A' && direction != 'D') ||
                (input == 'D' && direction != 'A')) {
            direction = input;
        }
    }

    public void doAction(float[] act) {
        char input = floatToAct(act);
        doAction(input);
    }

    public float reward() {
        Point head = snake.peekFirst();
        if(head == null) {
            return 0;
        }

        if (gameOver) {
            return ( - 2 -  1f / (WIDTH + HEIGHT) * head.dx(food));
        } else {
            return (eatFood - 2f / (WIDTH + HEIGHT) * head.dx(food));
        }
    }

    // Point
    private static class Point {
        int x, y;
        Point(int x, int y) { this.x = x; this.y = y; }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Point)) return false;
            Point p = (Point) o;
            return x == p.x && y == p.y;
        }

        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }

        public float dx(Point p){
            return  Math.abs(x - p.x) +  Math.abs(y - p.y) ;
        }
    }

    public static class BoardDisplay extends JFrame {

        private final JTextArea boardArea;

        public BoardDisplay() {
            setTitle("SnakeGame");
            setSize(400, 400);
            setDefaultCloseOperation(EXIT_ON_CLOSE);
            setLocationRelativeTo(null);

            boardArea = new JTextArea();
            boardArea.setFont(new Font("Courier New", Font.PLAIN, 20));
            boardArea.setEditable(false);

            JButton refreshButton = new JButton("test model");

            refreshButton.addActionListener(e -> updateBoard());
            JButton b2 = new JButton("　");
            b2.addActionListener(e -> f2());

            setLayout(new BorderLayout());
            add(new JScrollPane(boardArea), BorderLayout.CENTER);
            add(b2,BorderLayout.NORTH);
            add(refreshButton, BorderLayout.SOUTH);

        }

        private void updateBoard() {
            QNetworkTrainer.Test_model = !QNetworkTrainer.Test_model;
        }

        private void f2() {

        }

        public void  setText(String text) {
            StringBuilder result = new StringBuilder(" ");
            for (int i = 0; i < text.length(); i++) {
                result.append(text.charAt(i)).append(" ");
            }

            boardArea.setText(result.toString());
        }

    }

}