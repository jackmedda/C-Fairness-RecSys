/*    */ package org.gradle.wrapper;
/*    */ 
/*    */ import java.net.URI;
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
/*    */ public class WrapperConfiguration
/*    */ {
/*    */   private URI distribution;
/* 22 */   private String distributionBase = "GRADLE_USER_HOME";
/* 23 */   private String distributionPath = "wrapper/dists";
/* 24 */   private String zipBase = "GRADLE_USER_HOME";
/* 25 */   private String zipPath = "wrapper/dists";
/*    */   
/*    */   public URI getDistribution() {
/* 28 */     return this.distribution;
/*    */   }
/*    */   
/*    */   public void setDistribution(URI distribution) {
/* 32 */     this.distribution = distribution;
/*    */   }
/*    */   
/*    */   public String getDistributionBase() {
/* 36 */     return this.distributionBase;
/*    */   }
/*    */   
/*    */   public void setDistributionBase(String distributionBase) {
/* 40 */     this.distributionBase = distributionBase;
/*    */   }
/*    */   
/*    */   public String getDistributionPath() {
/* 44 */     return this.distributionPath;
/*    */   }
/*    */   
/*    */   public void setDistributionPath(String distributionPath) {
/* 48 */     this.distributionPath = distributionPath;
/*    */   }
/*    */   
/*    */   public String getZipBase() {
/* 52 */     return this.zipBase;
/*    */   }
/*    */   
/*    */   public void setZipBase(String zipBase) {
/* 56 */     this.zipBase = zipBase;
/*    */   }
/*    */   
/*    */   public String getZipPath() {
/* 60 */     return this.zipPath;
/*    */   }
/*    */   
/*    */   public void setZipPath(String zipPath) {
/* 64 */     this.zipPath = zipPath;
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\WrapperConfiguration.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */