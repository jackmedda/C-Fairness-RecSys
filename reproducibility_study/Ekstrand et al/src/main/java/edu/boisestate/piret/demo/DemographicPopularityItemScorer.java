package edu.boisestate.piret.demo;

import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonTypes;
import org.lenskit.data.entities.Entity;
import org.lenskit.results.Results;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Score items by demographic-stratified popularity.
 */
public class DemographicPopularityItemScorer extends AbstractItemScorer {
    private final DemographicPopularityModel model;
    private final DataAccessObject dao;

    @Inject
    public DemographicPopularityItemScorer(DemographicPopularityModel model, DataAccessObject dao) {
        this.model = model;
        this.dao = dao;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        Entity u = dao.lookupEntity(CommonTypes.USER, user);
        if (u == null) {
            return Results.newResultMap();
        }

        Long2DoubleMap pop = model.getUserStrataPopularities(u);
        List<Result> results = new ArrayList<>(items.size());
        for (long item: items) {
            double ipop = pop.get(item);
            results.add(Results.create(item, ipop));
        }
        return Results.newResultMap(results);
    }
}
