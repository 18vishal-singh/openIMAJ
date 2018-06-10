package test.imageProcessing;

import java.io.IOException;
import java.util.List;

import org.openimaj.image.FImage;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.colour.Transforms;
import org.openimaj.image.processing.face.detection.DetectedFace;
import org.openimaj.image.processing.face.detection.FaceDetector;
import org.openimaj.image.processing.face.detection.HaarCascadeDetector;
import org.openimaj.image.processing.face.detection.keypoints.FKEFaceDetector;
import org.openimaj.image.processing.face.detection.keypoints.KEDetectedFace;
import org.openimaj.video.VideoDisplay;
import org.openimaj.video.VideoDisplayListener;
import org.openimaj.video.capture.VideoCapture;

/**
 * This is used to detect faces from the web-cam
 * 
 */
public class App {

	public static void main(String[] args) throws IOException {
		final VideoCapture vc = new VideoCapture(320, 240);
		final VideoDisplay<MBFImage> vd = VideoDisplay.createVideoDisplay(vc);
		vd.addVideoListener(new VideoDisplayListener<MBFImage>() {
			public void beforeUpdate(MBFImage frame) {
				final HaarCascadeDetector detector = HaarCascadeDetector.BuiltInCascade.frontalface_alt2.load();
				FaceDetector<KEDetectedFace, FImage> fd = new FKEFaceDetector(detector);
				List<KEDetectedFace> faces = fd.detectFaces(Transforms.calculateIntensity(frame));

				for (final DetectedFace face : faces) {
					frame.drawShape(face.getBounds(), RGBColour.RED);

				}
			}

			public void afterUpdate(VideoDisplay<MBFImage> display) {
			}
		});
	}
}