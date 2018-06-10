package test.imageProcessing;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.List;

import javax.imageio.ImageIO;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.colour.Transforms;
import org.openimaj.image.processing.face.detection.DetectedFace;
import org.openimaj.image.processing.face.detection.FaceDetector;
import org.openimaj.image.processing.face.detection.HaarCascadeDetector;

public class FaceDetection {

	public static void main(String[] args) throws MalformedURLException, IOException {
		MBFImage img = ImageUtilities.readMBF(new File("data/obamaGroup.jpeg"));
		FaceDetector<DetectedFace, FImage> fd = new HaarCascadeDetector(40);
		List<DetectedFace> faces = fd.detectFaces(Transforms.calculateIntensity(img));
		int i = 1;
		for (DetectedFace face : faces) {
			img.drawShape(face.getBounds(), RGBColour.RED);
			String name = "output/sampleFaces/" + i + ".png";
			File outputFile = new File(name);
			try {
				ImageIO.write(ImageUtilities.createBufferedImageForDisplay(face.getFacePatch()), "png", outputFile);
			} catch (IOException e) {
				e.printStackTrace();
			}
			i++;
		}
		DisplayUtilities.display(img);
	}

}