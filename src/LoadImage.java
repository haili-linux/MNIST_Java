



import haili.deeplearn.BpNetwork;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Scanner;

public class LoadImage {

    /**
     * bmp图片像素转数组
     * @param imgsrc bmp图片
     * @return 。
     * @throws Exception null
     */
    public static float[] bmpToRgbList_L(BufferedImage imgsrc) {
        float[] rgb = new float[imgsrc.getWidth() * imgsrc.getHeight()];

        //创建一个灰度模式的图片
        BufferedImage back = new BufferedImage(imgsrc.getWidth(), imgsrc.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        int width = imgsrc.getWidth();
        int height = imgsrc.getHeight();

        int index = 0;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                back.setRGB(i, j, imgsrc.getRGB(i, j));
                rgb[index] = (back.getRGB(i,j) &  0xff) / 255.0f;
                index++;
            }
        }
        return rgb;
    }

    public static float[] bmpToRgbList_L(String filename) throws Exception {
        BufferedImage imgsrc = ImageIO.read(new File(filename));
        return  bmpToRgbList_L(imgsrc);
    }

}
