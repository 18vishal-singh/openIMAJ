package test.imageProcessing;

import java.io.File;
import java.io.IOException;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.FastBasicKeypointMatcher;
import org.openimaj.feature.local.matcher.consistent.ConsistentLocalFeatureMatcher2d;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.colour.Transforms;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.renderer.MBFImageRenderer;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.math.geometry.transforms.HomographyModel;
import org.openimaj.math.geometry.transforms.HomographyRefinement;
import org.openimaj.math.geometry.transforms.MatrixTransformProvider;
import org.openimaj.math.geometry.transforms.check.TransformMatrixConditionCheck;
import org.openimaj.math.geometry.transforms.estimation.RobustHomographyEstimator;
import org.openimaj.math.model.fit.RANSAC;
import org.openimaj.video.Video;
import org.openimaj.video.xuggle.XuggleVideo;

import Jama.Matrix;

/**
 * This class will detect the frames or pic provided from a video and if
 * matched, it will point out with making square brackets
 *
 */
public class ObjectDetection {

	public static void main(String args[]) throws IOException {
		new ObjectMainDetection();
	}
}

class ObjectMainDetection {
	private ConsistentLocalFeatureMatcher2d<Keypoint> matcher1;
	private ConsistentLocalFeatureMatcher2d<Keypoint> matcher2;
	private ConsistentLocalFeatureMatcher2d<Keypoint> matcher3;

	final DoGSIFTEngine engine;
	// private RenderMode renderMode = RenderMode.SQUARE;

	private MBFImage modelImage1, modelImage2, modelImage3;

	public ObjectMainDetection() throws IOException {
		this.engine = new DoGSIFTEngine();
		this.engine.getOptions().setDoubleInitialImage(true);
		this.matcher1 = new ConsistentLocalFeatureMatcher2d<Keypoint>(new FastBasicKeypointMatcher<Keypoint>(8));
		this.matcher2 = new ConsistentLocalFeatureMatcher2d<Keypoint>(new FastBasicKeypointMatcher<Keypoint>(8));
		this.matcher3 = new ConsistentLocalFeatureMatcher2d<Keypoint>(new FastBasicKeypointMatcher<Keypoint>(8));
		final RobustHomographyEstimator ransac = new RobustHomographyEstimator(0.5, 1500,
				new RANSAC.PercentageInliersStoppingCondition(0.6), HomographyRefinement.NONE,
				new TransformMatrixConditionCheck<HomographyModel>(10000));
		this.matcher1.setFittingModel(ransac);
		this.matcher2.setFittingModel(ransac);
		this.matcher3.setFittingModel(ransac);
		LoadReferenceObject();
		StartVideo();
	}

	public void StartVideo() throws IOException {
		Video<MBFImage> video = new XuggleVideo(new File("data/sample.mkv"));
		// Video<MBFImage> video = new XuggleVideo(new
		// URL("http://playertest.longtailvideo.com/adaptive/captions/playlist.m3u8"));
		// final Video<MBFImage> video = new VideoCapture(320, 240);
		for (MBFImage mbfImage : video) {
			final LocalFeatureList<Keypoint> kpl = this.engine
					.findFeatures(Transforms.calculateIntensityNTSC(mbfImage));
			final MBFImageRenderer renderer = mbfImage.createRenderer();
			renderer.drawPoints(kpl, RGBColour.MAGENTA, 3);

			if (this.matcher1.findMatches(kpl)
					&& ((MatrixTransformProvider) this.matcher1.getModel()).getTransform().cond() < 1e6) {
				try {
					final Matrix boundsToPoly = ((MatrixTransformProvider) this.matcher1.getModel()).getTransform()
							.inverse();
					if (modelImage1.getBounds().transform(boundsToPoly).isConvex()) {
						renderer.drawShape(this.modelImage1.getBounds().transform(boundsToPoly), 3, RGBColour.RED);
						renderer.drawText("bird 1",
								this.modelImage2.getBounds().transform(boundsToPoly).calculateCentroid(),
								HersheyFont.TIMES_BOLD, 15, RGBColour.BLACK);
					}
				} catch (Exception e) {
					System.out.println(e.getMessage());
				}

			}
			if (this.matcher2.findMatches(kpl)
					&& ((MatrixTransformProvider) this.matcher2.getModel()).getTransform().cond() < 1e6) {
				try {
					final Matrix boundsToPoly = ((MatrixTransformProvider) this.matcher2.getModel()).getTransform()
							.inverse();

					if (modelImage2.getBounds().transform(boundsToPoly).isConvex()) {
						renderer.drawShape(this.modelImage2.getBounds().transform(boundsToPoly), 3, RGBColour.RED);
						renderer.drawText("bird 2",
								this.modelImage2.getBounds().transform(boundsToPoly).calculateCentroid(),
								HersheyFont.TIMES_BOLD, 15, RGBColour.BLACK);
					}
				} catch (Exception e) {
					System.out.println(e.getMessage());
				}

			}
			if (this.matcher3.findMatches(kpl)
					&& ((MatrixTransformProvider) this.matcher3.getModel()).getTransform().cond() < 1e6) {
				try {
					final Matrix boundsToPoly = ((MatrixTransformProvider) this.matcher3.getModel()).getTransform()
							.inverse();
					if (modelImage3.getBounds().transform(boundsToPoly).isConvex()) {
						renderer.drawShape(this.modelImage3.getBounds().transform(boundsToPoly), 3, RGBColour.RED);
						renderer.drawText("grass",
								this.modelImage2.getBounds().transform(boundsToPoly).calculateCentroid(),
								HersheyFont.TIMES_BOLD, 15, RGBColour.BLACK);
					}
				} catch (Exception e) {
					System.out.println(e.getMessage());
				}

			}
			DisplayUtilities.displayName(mbfImage, "ObjectDetection");
		}
	}

	public void LoadReferenceObject() {

		final DoGSIFTEngine engine = new DoGSIFTEngine();
		engine.getOptions().setDoubleInitialImage(true);

		try {
			modelImage1 = ImageUtilities.readMBF(new File("data/bird.jpg"));
			modelImage2 = ImageUtilities.readMBF(new File("data/bird2.jpg"));
			modelImage3 = ImageUtilities.readMBF(new File("data/grass.jpg"));
		} catch (IOException e) {
			e.printStackTrace();
		}

		FImage modelF1 = Transforms.calculateIntensityNTSC(modelImage1);
		this.matcher1.setModelFeatures(engine.findFeatures(modelF1));

		FImage modelF2 = Transforms.calculateIntensityNTSC(modelImage2);
		this.matcher2.setModelFeatures(engine.findFeatures(modelF2));

		FImage modelF3 = Transforms.calculateIntensityNTSC(modelImage3);
		this.matcher3.setModelFeatures(engine.findFeatures(modelF3));

	}
}
