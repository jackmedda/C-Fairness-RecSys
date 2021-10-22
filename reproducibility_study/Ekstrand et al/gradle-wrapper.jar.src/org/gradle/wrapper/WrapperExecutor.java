/*     */ package org.gradle.wrapper;
/*     */ 
/*     */ import java.io.File;
/*     */ import java.io.FileInputStream;
/*     */ import java.io.IOException;
/*     */ import java.io.InputStream;
/*     */ import java.net.URI;
/*     */ import java.net.URISyntaxException;
/*     */ import java.util.Formatter;
/*     */ import java.util.Properties;
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
/*     */ public class WrapperExecutor
/*     */ {
/*     */   public static final String DISTRIBUTION_URL_PROPERTY = "distributionUrl";
/*     */   public static final String DISTRIBUTION_BASE_PROPERTY = "distributionBase";
/*     */   public static final String ZIP_STORE_BASE_PROPERTY = "zipStoreBase";
/*     */   public static final String DISTRIBUTION_PATH_PROPERTY = "distributionPath";
/*     */   public static final String ZIP_STORE_PATH_PROPERTY = "zipStorePath";
/*     */   private final Properties properties;
/*     */   private final File propertiesFile;
/*     */   private final Appendable warningOutput;
/*  36 */   private final WrapperConfiguration config = new WrapperConfiguration();
/*     */   
/*     */   public static WrapperExecutor forProjectDirectory(File projectDir, Appendable warningOutput) {
/*  39 */     return new WrapperExecutor(new File(projectDir, "gradle/wrapper/gradle-wrapper.properties"), new Properties(), warningOutput);
/*     */   }
/*     */   
/*     */   public static WrapperExecutor forWrapperPropertiesFile(File propertiesFile, Appendable warningOutput) {
/*  43 */     if (!propertiesFile.exists()) {
/*  44 */       throw new RuntimeException(String.format("Wrapper properties file '%s' does not exist.", new Object[] { propertiesFile }));
/*     */     }
/*  46 */     return new WrapperExecutor(propertiesFile, new Properties(), warningOutput);
/*     */   }
/*     */   
/*     */   WrapperExecutor(File propertiesFile, Properties properties, Appendable warningOutput) {
/*  50 */     this.properties = properties;
/*  51 */     this.propertiesFile = propertiesFile;
/*  52 */     this.warningOutput = warningOutput;
/*  53 */     if (propertiesFile.exists()) {
/*     */       try {
/*  55 */         loadProperties(propertiesFile, properties);
/*  56 */         this.config.setDistribution(prepareDistributionUri());
/*  57 */         this.config.setDistributionBase(getProperty("distributionBase", this.config.getDistributionBase()));
/*  58 */         this.config.setDistributionPath(getProperty("distributionPath", this.config.getDistributionPath()));
/*  59 */         this.config.setZipBase(getProperty("zipStoreBase", this.config.getZipBase()));
/*  60 */         this.config.setZipPath(getProperty("zipStorePath", this.config.getZipPath()));
/*  61 */       } catch (Exception e) {
/*  62 */         throw new RuntimeException(String.format("Could not load wrapper properties from '%s'.", new Object[] { propertiesFile }), e);
/*     */       } 
/*     */     }
/*     */   }
/*     */   
/*     */   private URI prepareDistributionUri() throws URISyntaxException {
/*  68 */     URI source = readDistroUrl();
/*  69 */     if (source.getScheme() == null)
/*     */     {
/*  71 */       return (new File(this.propertiesFile.getParentFile(), source.getSchemeSpecificPart())).toURI();
/*     */     }
/*  73 */     return source;
/*     */   }
/*     */ 
/*     */   
/*     */   private URI readDistroUrl() throws URISyntaxException {
/*  78 */     if (this.properties.getProperty("distributionUrl") != null) {
/*  79 */       return new URI(getProperty("distributionUrl"));
/*     */     }
/*     */     
/*  82 */     return readDistroUrlDeprecatedWay();
/*     */   }
/*     */   
/*     */   private URI readDistroUrlDeprecatedWay() throws URISyntaxException {
/*  86 */     String distroUrl = null;
/*     */     try {
/*  88 */       distroUrl = getProperty("urlRoot") + "/" + getProperty("distributionName") + "-" + getProperty("distributionVersion") + "-" + getProperty("distributionClassifier") + ".zip";
/*     */ 
/*     */ 
/*     */       
/*  92 */       Formatter formatter = new Formatter();
/*  93 */       formatter.format("Wrapper properties file '%s' contains deprecated entries 'urlRoot', 'distributionName', 'distributionVersion' and 'distributionClassifier'. These will be removed soon. Please use '%s' instead.%n", new Object[] { this.propertiesFile, "distributionUrl" });
/*  94 */       this.warningOutput.append(formatter.toString());
/*  95 */     } catch (Exception e) {
/*     */       
/*  97 */       reportMissingProperty("distributionUrl");
/*     */     } 
/*  99 */     return new URI(distroUrl);
/*     */   }
/*     */   
/*     */   private static void loadProperties(File propertiesFile, Properties properties) throws IOException {
/* 103 */     InputStream inStream = new FileInputStream(propertiesFile);
/*     */     try {
/* 105 */       properties.load(inStream);
/*     */     } finally {
/* 107 */       inStream.close();
/*     */     } 
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public URI getDistribution() {
/* 115 */     return this.config.getDistribution();
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   public WrapperConfiguration getConfiguration() {
/* 122 */     return this.config;
/*     */   }
/*     */   
/*     */   public void execute(String[] args, Install install, BootstrapMainStarter bootstrapMainStarter) throws Exception {
/* 126 */     File gradleHome = install.createDist(this.config);
/* 127 */     bootstrapMainStarter.start(args, gradleHome);
/*     */   }
/*     */   
/*     */   private String getProperty(String propertyName) {
/* 131 */     return getProperty(propertyName, null);
/*     */   }
/*     */   
/*     */   private String getProperty(String propertyName, String defaultValue) {
/* 135 */     String value = this.properties.getProperty(propertyName);
/* 136 */     if (value != null) {
/* 137 */       return value;
/*     */     }
/* 139 */     if (defaultValue != null) {
/* 140 */       return defaultValue;
/*     */     }
/* 142 */     return reportMissingProperty(propertyName);
/*     */   }
/*     */   
/*     */   private String reportMissingProperty(String propertyName) {
/* 146 */     throw new RuntimeException(String.format("No value with key '%s' specified in wrapper properties file '%s'.", new Object[] { propertyName, this.propertiesFile }));
/*     */   }
/*     */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\WrapperExecutor.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */