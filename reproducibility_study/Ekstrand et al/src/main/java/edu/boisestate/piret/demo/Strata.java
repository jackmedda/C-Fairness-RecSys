package edu.boisestate.piret.demo;

import org.lenskit.inject.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;
import java.util.List;

/**
 * Parameter controlling the user attributes on which to stratify popularity.
 */
@Documented
@Qualifier
@Parameter(String.class)
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.PARAMETER)
public @interface Strata {
}
