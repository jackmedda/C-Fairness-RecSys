import org.lenskit.api.ItemScorer
import org.lenskit.api.RatingPredictor
import org.lenskit.data.entities.CommonTypes
import org.lenskit.data.ratings.EntityCountRatingVectorPDAO
import org.lenskit.data.ratings.InteractionEntityType
import org.lenskit.data.ratings.RatingVectorPDAO
import org.lenskit.knn.NeighborhoodSize
import org.lenskit.knn.user.SimilaritySumUserNeighborhoodScorer
import org.lenskit.knn.user.UserNeighborhoodScorer
import org.lenskit.knn.user.UserUserItemScorer

bind ItemScorer to UserUserItemScorer

bind RatingVectorPDAO to EntityCountRatingVectorPDAO
set InteractionEntityType to CommonTypes.RATING

bind UserNeighborhoodScorer to SimilaritySumUserNeighborhoodScorer

bind RatingPredictor to null

set NeighborhoodSize to 30