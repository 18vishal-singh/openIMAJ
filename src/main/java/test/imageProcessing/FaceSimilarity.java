package test.imageProcessing;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Map;
import java.util.Map.Entry;

import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.face.detection.HaarCascadeDetector;
import org.openimaj.image.processing.face.detection.keypoints.FKEFaceDetector;
import org.openimaj.image.processing.face.detection.keypoints.KEDetectedFace;
import org.openimaj.image.processing.face.feature.FacePatchFeature;
import org.openimaj.image.processing.face.feature.FacePatchFeature.Extractor;
import org.openimaj.image.processing.face.feature.comparison.FaceFVComparator;
import org.openimaj.image.processing.face.similarity.FaceSimilarityEngine;
import org.openimaj.math.geometry.shape.Rectangle;

/**
 * This class is comparing faces from given sample to a group photo using
 * {@link HaarCascadeDetector}
 *
 */
public class FaceSimilarity {

	public static void main(String[] args) throws IOException {
		// first, we load two images
		final URL image1url = new URL(
				"http://s3.amazonaws.com/rapgenius/fema_-_39841_-_official_portrait_of_president-elect_barack_obama_on_jan-_13.jpg");

		final FImage image1 = ImageUtilities.readF(image1url);
		// final FImage image1 = ImageUtilities.readF(new File("data/obama.png"));
		// final FImage image2 = ImageUtilities.readF(image2url);
		// final FImage image1 = ImageUtilities.readF(new File("data/man1.png"));
		final FImage image2 = ImageUtilities.readF(new File("data/obamaGroup.jpeg"));

		final HaarCascadeDetector detector = HaarCascadeDetector.BuiltInCascade.frontalface_alt2.load();
		final FKEFaceDetector kedetector = new FKEFaceDetector(detector);
		final Extractor extractor = new FacePatchFeature.Extractor();
		final FaceFVComparator<FacePatchFeature, FloatFV> comparator = new FaceFVComparator<FacePatchFeature, FloatFV>(
				FloatFVComparison.EUCLIDEAN);
		final FaceSimilarityEngine<KEDetectedFace, FacePatchFeature, FImage> engine = new FaceSimilarityEngine<KEDetectedFace, FacePatchFeature, FImage>(
				kedetector, extractor, comparator);

		engine.setQuery(image1, "image1");
		// engine.setQuery(image12, "image12");
		engine.setTest(image2, "image2");

		// and then to do its work of detecting, extracting and comparing
		engine.performTest();

		// The following loop goes through the map of
		// each face in the first image to all the faces in the second:
		for (final Entry<String, Map<String, Double>> e : engine.getSimilarityDictionary().entrySet()) {
			// this computes the matching face in the second image with the
			// smallest distance:
			double bestScore = Double.MAX_VALUE;
			String best = null;
			for (final Entry<String, Double> matches : e.getValue().entrySet()) {
				if (matches.getValue() < bestScore) {
					bestScore = matches.getValue();
					best = matches.getKey();
				}
			}

			final FImage img = new FImage(image1.width + image2.width, Math.max(image1.height, image2.height));
			img.drawImage(image1, 0, 0);
			img.drawImage(image2, image1.width, 0);

			img.drawShape(engine.getBoundingBoxes().get(e.getKey()), 1F);

			final Rectangle r = engine.getBoundingBoxes().get(best);
			r.translate(image1.width, 0);
			img.drawShape(r, 1F);
			DisplayUtilities.display(img);
		}
	}
}
