/*    */ package org.gradle.wrapper;
/*    */ 
/*    */ import java.io.File;
/*    */ import java.lang.reflect.Method;
/*    */ import java.net.URL;
/*    */ import java.net.URLClassLoader;
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
/*    */ public class BootstrapMainStarter
/*    */ {
/*    */   public void start(String[] args, File gradleHome) throws Exception {
/* 25 */     File gradleJar = findLauncherJar(gradleHome);
/* 26 */     URLClassLoader contextClassLoader = new URLClassLoader(new URL[] { gradleJar.toURI().toURL() }, ClassLoader.getSystemClassLoader().getParent());
/* 27 */     Thread.currentThread().setContextClassLoader(contextClassLoader);
/* 28 */     Class<?> mainClass = contextClassLoader.loadClass("org.gradle.launcher.GradleMain");
/* 29 */     Method mainMethod = mainClass.getMethod("main", new Class[] { String[].class });
/* 30 */     mainMethod.invoke(null, new Object[] { args });
/*    */   }
/*    */   
/*    */   private File findLauncherJar(File gradleHome) {
/* 34 */     for (File file : (new File(gradleHome, "lib")).listFiles()) {
/* 35 */       if (file.getName().matches("gradle-launcher-.*\\.jar")) {
/* 36 */         return file;
/*    */       }
/*    */     } 
/* 39 */     throw new RuntimeException(String.format("Could not locate the Gradle launcher JAR in Gradle distribution '%s'.", new Object[] { gradleHome }));
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\BootstrapMainStarter.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */