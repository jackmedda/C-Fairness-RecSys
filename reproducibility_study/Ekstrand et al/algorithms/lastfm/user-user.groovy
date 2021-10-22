import edu.boisestate.piret.demo.LogVectorNormalizer
import org.grouplens.lenskit.transform.threshold.RealThreshold
import org.grouplens.lenskit.transform.threshold.Threshold
import org.lenskit.api.ItemScorer
import org.lenskit.api.RatingPredictor
import org.lenskit.basic.SimpleCachingItemScorer
import org.lenskit.data.entities.EntityType
import org.lenskit.data.ratings.CountSumRatingVectorPDAO
import org.lenskit.data.ratings.EntityCountRatingVectorPDAO
import org.lenskit.data.ratings.InteractionEntityType
import org.lenskit.data.ratings.RatingVectorPDAO
import org.lenskit.knn.NeighborhoodSize
import org.lenskit.knn.user.SimilaritySumUserNeighborhoodScorer
import org.lenskit.knn.user.UserNeighborhoodScorer
import org.lenskit.knn.user.UserSimilarityThreshold
import org.lenskit.knn.user.UserUserItemScorer
import org.lenskit.transform.normalize.UserVectorNormalizer
import org.lenskit.transform.normalize.VectorNormalizer

bind ItemScorer to SimpleCachingItemScorer
within (SimpleCachingItemScorer) {
    bind ItemScorer to UserUserItemScorer
}

set InteractionEntityType to EntityType.forName("artist-count")

bind UserNeighborhoodScorer to SimilaritySumUserNeighborhoodScorer

bind RatingPredictor to null

set NeighborhoodSize to 30
bind (UserSimilarityThreshold, Threshold) to RealThreshold

algorithm("UU-C") {
    bind RatingVectorPDAO to CountSumRatingVectorPDAO
    within (UserVectorNormalizer) {
        bind VectorNormalizer to LogVectorNormalizer
    }
}

algorithm("UU-B") {
    bind RatingVectorPDAO to EntityCountRatingVectorPDAO
}