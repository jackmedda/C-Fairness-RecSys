import org.lenskit.api.ItemScorer
import org.lenskit.api.RatingPredictor
import org.lenskit.basic.PopularityRankItemScorer
import org.lenskit.data.entities.EntityType
import org.lenskit.data.ratings.InteractionEntityType

bind ItemScorer to PopularityRankItemScorer
bind RatingPredictor to null
set InteractionEntityType to EntityType.forName("artist-count")
