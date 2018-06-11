package test.imageProcessing;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.colour.Transforms;
import org.openimaj.image.model.EigenImages;
import org.openimaj.image.processing.face.detection.DetectedFace;
import org.openimaj.image.processing.face.detection.FaceDetector;
import org.openimaj.image.processing.face.detection.HaarCascadeDetector;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * Matching faces from an image from our stored dataset
 * 
 * @author vishal singh
 *
 */
public class Main {

	public static void main(String[] args) throws MalformedURLException, IOException {
		final int nEigenvectors = 100;
		final EigenImages eigen = new EigenImages(nEigenvectors);
		final VFSGroupDataset<FImage> dataset = new VFSGroupDataset<FImage>(
				"C:\\Users\\599763\\Desktop\\code-hub\\imageProcessing\\data\\dataSet", ImageUtilities.FIMAGE_READER);

		List<FImage> basisImages = new ArrayList<FImage>();
		;
		for (final Entry<String, VFSListDataset<FImage>> training : dataset.entrySet()) {
			basisImages.addAll(training.getValue());
		}
		eigen.train(basisImages);
		final List<FImage> eigenFaces = new ArrayList<FImage>();
		for (int i = 0; i < 19; i++) {
			eigenFaces.add(eigen.visualisePC(i));
		}
		DisplayUtilities.display("EigenFaces", eigenFaces);

		/*
		 * Build a map of person->[features] for all the training data
		 */
		System.out.println("Checkpoint");
		final Map<String, DoubleFV[]> features = new HashMap<String, DoubleFV[]>();
		for (final Entry<String, VFSListDataset<FImage>> person : dataset.entrySet()) {
			final DoubleFV[] fvs = new DoubleFV[10];
			for (int i = 0; i < 10; i++) {
				final FImage face = person.getValue().get(i);
				fvs[i] = eigen.extractFeature(face);
			}
			features.put(person.getKey(), fvs);
		}
		System.out.println("total person-" + features.size());
		
		/*
		 * Extracting faces from image and comparing
		 */
		MBFImage img = ImageUtilities.readMBF(new File("data/obamaf.jpg"));
		FaceDetector<DetectedFace, FImage> fd = new HaarCascadeDetector(40);
		List<DetectedFace> faces = fd.detectFaces(Transforms.calculateIntensity(img));
		DetectedFace tempFace=null;
		for (DetectedFace face : faces) {
			FImage testFace = ResizeProcessor.resample(face.getFacePatch(), 87, 112);
			final DoubleFV testFeature = eigen.extractFeature(testFace);
			
			String bestPerson = null;
			double minDistance = Double.MAX_VALUE;
			for (final String person : features.keySet()) {
				System.out.println("THis is---------------------"+person);
				for (final DoubleFV fv : features.get(person)) {
					final double distance = fv.compare(testFeature, DoubleFVComparison.EUCLIDEAN);

					if (distance < minDistance) {
						minDistance = distance;
						bestPerson = person;
					}
				}
			}
			System.out.println(minDistance);
			System.out.println("guess: " + bestPerson);
			
			if(minDistance> 12)
				img.drawShape(face.getBounds(), RGBColour.RED);
		}
		DisplayUtilities.display(img);
	}


}
