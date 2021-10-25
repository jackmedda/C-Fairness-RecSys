/*     */ package org.gradle.wrapper;
/*     */ 
/*     */ import java.io.File;
/*     */ import java.io.InputStream;
/*     */ import java.net.URI;
/*     */ import java.net.URISyntaxException;
/*     */ import java.util.HashMap;
/*     */ import java.util.Properties;
/*     */ import org.gradle.cli.CommandLineParser;
/*     */ import org.gradle.cli.ParsedCommandLine;
/*     */ import org.gradle.cli.SystemPropertiesCommandLineConverter;
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
/*     */ public class GradleWrapperMain
/*     */ {
/*     */   public static final String GRADLE_USER_HOME_OPTION = "g";
/*     */   public static final String GRADLE_USER_HOME_DETAILED_OPTION = "gradle-user-home";
/*     */   
/*     */   public static void main(String[] args) throws Exception {
/*  35 */     File wrapperJar = wrapperJar();
/*  36 */     File propertiesFile = wrapperProperties(wrapperJar);
/*  37 */     File rootDir = rootDir(wrapperJar);
/*     */     
/*  39 */     CommandLineParser parser = new CommandLineParser();
/*  40 */     parser.allowUnknownOptions();
/*  41 */     parser.option(new String[] { "g", "gradle-user-home" }).hasArgument();
/*     */     
/*  43 */     SystemPropertiesCommandLineConverter converter = new SystemPropertiesCommandLineConverter();
/*  44 */     converter.configure(parser);
/*     */     
/*  46 */     ParsedCommandLine options = parser.parse(args);
/*     */     
/*  48 */     Properties systemProperties = System.getProperties();
/*  49 */     systemProperties.putAll(converter.convert(options, new HashMap<Object, Object>()));
/*     */     
/*  51 */     File gradleUserHome = gradleUserHome(options);
/*     */     
/*  53 */     addSystemProperties(gradleUserHome, rootDir);
/*     */     
/*  55 */     WrapperExecutor wrapperExecutor = WrapperExecutor.forWrapperPropertiesFile(propertiesFile, System.out);
/*  56 */     wrapperExecutor.execute(args, new Install(new Download("gradlew", wrapperVersion()), new PathAssembler(gradleUserHome)), new BootstrapMainStarter());
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   private static void addSystemProperties(File gradleHome, File rootDir) {
/*  63 */     System.getProperties().putAll(SystemPropertiesHandler.getSystemProperties(new File(gradleHome, "gradle.properties")));
/*  64 */     System.getProperties().putAll(SystemPropertiesHandler.getSystemProperties(new File(rootDir, "gradle.properties")));
/*     */   }
/*     */   
/*     */   private static File rootDir(File wrapperJar) {
/*  68 */     return wrapperJar.getParentFile().getParentFile().getParentFile();
/*     */   }
/*     */   
/*     */   private static File wrapperProperties(File wrapperJar) {
/*  72 */     return new File(wrapperJar.getParent(), wrapperJar.getName().replaceFirst("\\.jar$", ".properties"));
/*     */   }
/*     */   
/*     */   private static File wrapperJar() {
/*     */     URI location;
/*     */     try {
/*  78 */       location = GradleWrapperMain.class.getProtectionDomain().getCodeSource().getLocation().toURI();
/*  79 */     } catch (URISyntaxException e) {
/*  80 */       throw new RuntimeException(e);
/*     */     } 
/*  82 */     if (!location.getScheme().equals("file")) {
/*  83 */       throw new RuntimeException(String.format("Cannot determine classpath for wrapper Jar from codebase '%s'.", new Object[] { location }));
/*     */     }
/*  85 */     return new File(location.getPath());
/*     */   }
/*     */   
/*     */   static String wrapperVersion() {
/*     */     try {
/*  90 */       InputStream resourceAsStream = GradleWrapperMain.class.getResourceAsStream("/build-receipt.properties");
/*  91 */       if (resourceAsStream == null) {
/*  92 */         throw new RuntimeException("No build receipt resource found.");
/*     */       }
/*  94 */       Properties buildReceipt = new Properties();
/*     */       try {
/*  96 */         buildReceipt.load(resourceAsStream);
/*  97 */         String versionNumber = buildReceipt.getProperty("versionNumber");
/*  98 */         if (versionNumber == null) {
/*  99 */           throw new RuntimeException("No version number specified in build receipt resource.");
/*     */         }
/* 101 */         return versionNumber;
/*     */       } finally {
/* 103 */         resourceAsStream.close();
/*     */       } 
/* 105 */     } catch (Exception e) {
/* 106 */       throw new RuntimeException("Could not determine wrapper version.", e);
/*     */     } 
/*     */   }
/*     */   
/*     */   private static File gradleUserHome(ParsedCommandLine options) {
/* 111 */     if (options.hasOption("g")) {
/* 112 */       return new File(options.option("g").getValue());
/*     */     }
/* 114 */     return GradleUserHomeLookup.gradleUserHome();
/*     */   }
/*     */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\GradleWrapperMain.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */