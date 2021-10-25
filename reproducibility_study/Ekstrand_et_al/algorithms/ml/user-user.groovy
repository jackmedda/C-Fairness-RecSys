import org.lenskit.api.ItemScorer
import org.lenskit.baseline.BaselineScorer
import org.lenskit.basic.FallbackItemScorer
import org.lenskit.basic.TopNItemRecommender
import org.lenskit.bias.BiasItemScorer
import org.lenskit.bias.BiasModel
import org.lenskit.bias.UserItemBiasModel
import org.lenskit.knn.NeighborhoodSize
import org.lenskit.knn.user.UserUserItemScorer
import org.lenskit.transform.normalize.MeanCenteringVectorNormalizer
import org.lenskit.transform.normalize.UserVectorNormalizer
import org.lenskit.transform.normalize.VectorNormalizer

bind ItemScorer to UserUserItemScorer
within (UserVectorNormalizer) {
    bind VectorNormalizer to MeanCenteringVectorNormalizer
}
bind (BaselineScorer, ItemScorer) to BiasItemScorer
bind BiasModel to UserItemBiasModel
at (TopNItemRecommender) {
    bind ItemScorer to FallbackItemScorer
}

set NeighborhoodSize to 30