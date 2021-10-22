import org.lenskit.api.ItemScorer
import org.lenskit.api.RatingPredictor
import org.lenskit.baseline.BaselineScorer
import org.lenskit.basic.FallbackItemScorer
import org.lenskit.basic.TopNItemRecommender
import org.lenskit.bias.BiasItemScorer
import org.lenskit.bias.BiasModel
import org.lenskit.bias.ItemBiasModel
import org.lenskit.bias.UserItemBiasModel
import org.lenskit.data.entities.CommonTypes
import org.lenskit.data.ratings.EntityCountRatingVectorPDAO
import org.lenskit.data.ratings.InteractionEntityType
import org.lenskit.data.ratings.RatingVectorPDAO
import org.lenskit.knn.NeighborhoodSize
import org.lenskit.knn.item.ItemItemScorer
import org.lenskit.knn.item.ModelSize
import org.lenskit.knn.item.NeighborhoodScorer
import org.lenskit.knn.item.SimilaritySumNeighborhoodScorer
import org.lenskit.transform.normalize.BiasUserVectorNormalizer
import org.lenskit.transform.normalize.UserVectorNormalizer

bind ItemScorer to ItemItemScorer
bind NeighborhoodScorer to SimilaritySumNeighborhoodScorer
bind RatingVectorPDAO to EntityCountRatingVectorPDAO
set InteractionEntityType to CommonTypes.RATING

bind RatingPredictor to null

set NeighborhoodSize to 20
set ModelSize to 4000