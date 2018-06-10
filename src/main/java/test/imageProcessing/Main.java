package test.imageProcessing;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.model.EigenImages;

public class Main {

	public static void main(String[] args) throws IOException {
		final int nEigenvectors = 100;
		final EigenImages eigen = new EigenImages(nEigenvectors);
		final VFSGroupDataset<FImage> dataset = new VFSGroupDataset<FImage>(
				"F:\\eclipse workspace(banglore)\\imageProcessing\\data\\att_faces", ImageUtilities.FIMAGE_READER);

		for (final Entry<String, VFSListDataset<FImage>> training : dataset.entrySet()) {
			final List<FImage> basisImages = training.getValue();
			eigen.train(basisImages);

			// DisplayUtilities.display(training.getKey(), training.getValue());
		}
		final List<FImage> eigenFaces = new ArrayList<FImage>();
		for (int i = 0; i < 9; i++) {
			eigenFaces.add(eigen.visualisePC(i));
		}
		DisplayUtilities.display("EigenFaces", eigenFaces);

		/*
		 * Build a map of person->[features] for all the training data
		 */
		final Map<String, DoubleFV[]> features = new HashMap<String, DoubleFV[]>();
		for (final Entry<String, VFSListDataset<FImage>> person : dataset.entrySet()) {
			final DoubleFV[] fvs = new DoubleFV[5];

			for (int i = 0; i < 10; i++) {
				final FImage face = person.getValue().get(i);
				fvs[i] = eigen.extractFeature(face);
			}
			features.put(person.getKey(), fvs);
		}
		for (Map.Entry m : features.entrySet()) {
			System.out.println(m.getKey() + " " + m.getValue());
		}
	}

}
