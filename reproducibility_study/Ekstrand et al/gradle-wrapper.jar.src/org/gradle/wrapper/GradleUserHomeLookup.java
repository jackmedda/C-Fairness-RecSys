/*    */ package org.gradle.wrapper;
/*    */ 
/*    */ import java.io.File;
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ public class GradleUserHomeLookup
/*    */ {
/* 22 */   public static final String DEFAULT_GRADLE_USER_HOME = System.getProperty("user.home") + "/.gradle";
/*    */   public static final String GRADLE_USER_HOME_PROPERTY_KEY = "gradle.user.home";
/*    */   public static final String GRADLE_USER_HOME_ENV_KEY = "GRADLE_USER_HOME";
/*    */   
/*    */   public static File gradleUserHome() {
/*    */     String gradleUserHome;
/* 28 */     if ((gradleUserHome = System.getProperty("gradle.user.home")) != null) {
/* 29 */       return new File(gradleUserHome);
/*    */     }
/* 31 */     if ((gradleUserHome = System.getenv("GRADLE_USER_HOME")) != null) {
/* 32 */       return new File(gradleUserHome);
/*    */     }
/* 34 */     return new File(DEFAULT_GRADLE_USER_HOME);
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\GradleUserHomeLookup.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */