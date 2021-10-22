/*     */ package org.gradle.wrapper;
/*     */ 
/*     */ import java.io.File;
/*     */ import java.math.BigInteger;
/*     */ import java.net.URI;
/*     */ import java.security.MessageDigest;
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ public class PathAssembler
/*     */ {
/*     */   public static final String GRADLE_USER_HOME_STRING = "GRADLE_USER_HOME";
/*     */   public static final String PROJECT_STRING = "PROJECT";
/*     */   private File gradleUserHome;
/*     */   
/*     */   public PathAssembler() {}
/*     */   
/*     */   public PathAssembler(File gradleUserHome) {
/*  33 */     this.gradleUserHome = gradleUserHome;
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public LocalDistribution getDistribution(WrapperConfiguration configuration) {
/*  40 */     String baseName = getDistName(configuration.getDistribution());
/*  41 */     String distName = removeExtension(baseName);
/*  42 */     String rootDirName = rootDirName(distName, configuration);
/*  43 */     File distDir = new File(getBaseDir(configuration.getDistributionBase()), configuration.getDistributionPath() + "/" + rootDirName);
/*  44 */     File distZip = new File(getBaseDir(configuration.getZipBase()), configuration.getZipPath() + "/" + rootDirName + "/" + baseName);
/*  45 */     return new LocalDistribution(distDir, distZip);
/*     */   }
/*     */   
/*     */   private String rootDirName(String distName, WrapperConfiguration configuration) {
/*  49 */     String urlHash = getMd5Hash(configuration.getDistribution().toString());
/*  50 */     return String.format("%s/%s", new Object[] { distName, urlHash });
/*     */   }
/*     */   
/*     */   private String getMd5Hash(String string) {
/*     */     try {
/*  55 */       MessageDigest messageDigest = MessageDigest.getInstance("MD5");
/*  56 */       byte[] bytes = string.getBytes();
/*  57 */       messageDigest.update(bytes);
/*  58 */       return (new BigInteger(1, messageDigest.digest())).toString(32);
/*  59 */     } catch (Exception e) {
/*  60 */       throw new RuntimeException("Could not hash input string.", e);
/*     */     } 
/*     */   }
/*     */   
/*     */   private String removeExtension(String name) {
/*  65 */     int p = name.lastIndexOf(".");
/*  66 */     if (p < 0) {
/*  67 */       return name;
/*     */     }
/*  69 */     return name.substring(0, p);
/*     */   }
/*     */   
/*     */   private String getDistName(URI distUrl) {
/*  73 */     String path = distUrl.getPath();
/*  74 */     int p = path.lastIndexOf("/");
/*  75 */     if (p < 0) {
/*  76 */       return path;
/*     */     }
/*  78 */     return path.substring(p + 1);
/*     */   }
/*     */   
/*     */   private File getBaseDir(String base) {
/*  82 */     if (base.equals("GRADLE_USER_HOME"))
/*  83 */       return this.gradleUserHome; 
/*  84 */     if (base.equals("PROJECT")) {
/*  85 */       return new File(System.getProperty("user.dir"));
/*     */     }
/*  87 */     throw new RuntimeException("Base: " + base + " is unknown");
/*     */   }
/*     */   
/*     */   public class LocalDistribution
/*     */   {
/*     */     private final File distZip;
/*     */     private final File distDir;
/*     */     
/*     */     public LocalDistribution(File distDir, File distZip) {
/*  96 */       this.distDir = distDir;
/*  97 */       this.distZip = distZip;
/*     */     }
/*     */ 
/*     */ 
/*     */ 
/*     */     
/*     */     public File getDistributionDir() {
/* 104 */       return this.distDir;
/*     */     }
/*     */ 
/*     */ 
/*     */ 
/*     */     
/*     */     public File getZipFile() {
/* 111 */       return this.distZip;
/*     */     }
/*     */   }
/*     */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\PathAssembler.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */