import edu.boisestate.piret.demo.LogVectorNormalizer
import org.lenskit.api.ItemScorer
import org.lenskit.api.RatingPredictor
import org.lenskit.basic.PopularityRankItemScorer
import org.lenskit.data.entities.EntityType
import org.lenskit.data.ratings.InteractionEntityType
import org.lenskit.data.ratings.InteractionStatistics
import org.lenskit.transform.normalize.VectorNormalizer

bind ItemScorer to PopularityRankItemScorer
bind RatingPredictor to null
set InteractionEntityType to EntityType.forName("artist-count")
bind VectorNormalizer to LogVectorNormalizer
bind InteractionStatistics toProvider InteractionStatistics.CountSumISProvider