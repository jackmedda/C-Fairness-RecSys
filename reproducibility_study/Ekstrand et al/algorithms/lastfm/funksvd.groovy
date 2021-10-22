import edu.boisestate.piret.demo.LogCountRatingVectorPDAO
import org.grouplens.lenskit.iterative.IterationCount
import org.lenskit.api.ItemScorer
import org.lenskit.api.RatingPredictor
import org.lenskit.bias.BiasModel
import org.lenskit.bias.GlobalBiasModel
import org.lenskit.data.entities.EntityType
import org.lenskit.data.ratings.EntityCountRatingVectorPDAO
import org.lenskit.data.ratings.InteractionEntityType
import org.lenskit.data.ratings.RatingVectorPDAO
import org.lenskit.mf.funksvd.FeatureCount
import org.lenskit.mf.funksvd.FunkSVDItemScorer

bind ItemScorer to FunkSVDItemScorer
set FeatureCount to 40
set IterationCount to 150

set InteractionEntityType to EntityType.forName("artist-count")

bind RatingPredictor to null

bind BiasModel to new GlobalBiasModel(0)

algorithm("MF-C") {
    bind RatingVectorPDAO to LogCountRatingVectorPDAO
}

algorithm("MF-B") {
    bind RatingVectorPDAO to EntityCountRatingVectorPDAO
}