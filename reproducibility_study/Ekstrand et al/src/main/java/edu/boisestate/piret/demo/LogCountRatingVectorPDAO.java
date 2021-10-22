package edu.boisestate.piret.demo;

import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import org.lenskit.data.ratings.CountSumRatingVectorPDAO;
import org.lenskit.data.ratings.RatingVectorPDAO;
import org.lenskit.transform.normalize.VectorNormalizer;
import org.lenskit.util.IdBox;
import org.lenskit.util.io.ObjectStream;
import org.lenskit.util.io.ObjectStreams;

import javax.annotation.Nonnull;
import javax.inject.Inject;

public class LogCountRatingVectorPDAO implements RatingVectorPDAO {
    private final RatingVectorPDAO delegate;
    private final VectorNormalizer normalizer;

    @Inject
    public LogCountRatingVectorPDAO(CountSumRatingVectorPDAO dlg, LogVectorNormalizer norm) {
        delegate = dlg;
        normalizer = norm;
    }

    @Nonnull
    @Override
    public Long2DoubleMap userRatingVector(long user) {
        Long2DoubleMap vec = delegate.userRatingVector(user);
        return normalizer.makeTransformation(vec).apply(vec);
    }

    @Override
    public ObjectStream<IdBox<Long2DoubleMap>> streamUsers() {
        return ObjectStreams.transform(delegate.streamUsers(),
                                       (vec) -> IdBox.create(vec.getId(), normalizer.makeTransformation(vec.getValue())
                                                                                    .apply(vec.getValue())));
    }
}
