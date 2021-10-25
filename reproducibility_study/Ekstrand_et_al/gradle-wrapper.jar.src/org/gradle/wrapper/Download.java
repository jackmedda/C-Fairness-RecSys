/*    */ package org.gradle.wrapper;
/*    */ 
/*    */ import java.io.BufferedOutputStream;
/*    */ import java.io.File;
/*    */ import java.io.FileOutputStream;
/*    */ import java.io.InputStream;
/*    */ import java.io.OutputStream;
/*    */ import java.net.Authenticator;
/*    */ import java.net.PasswordAuthentication;
/*    */ import java.net.URI;
/*    */ import java.net.URL;
/*    */ import java.net.URLConnection;
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ public class Download
/*    */   implements IDownload
/*    */ {
/*    */   private static final int PROGRESS_CHUNK = 20000;
/*    */   private static final int BUFFER_SIZE = 10000;
/*    */   private final String applicationName;
/*    */   private final String applicationVersion;
/*    */   
/*    */   public Download(String applicationName, String applicationVersion) {
/* 29 */     this.applicationName = applicationName;
/* 30 */     this.applicationVersion = applicationVersion;
/* 31 */     configureProxyAuthentication();
/*    */   }
/*    */   
/*    */   private void configureProxyAuthentication() {
/* 35 */     if (System.getProperty("http.proxyUser") != null) {
/* 36 */       Authenticator.setDefault(new SystemPropertiesProxyAuthenticator());
/*    */     }
/*    */   }
/*    */   
/*    */   public void download(URI address, File destination) throws Exception {
/* 41 */     destination.getParentFile().mkdirs();
/* 42 */     downloadInternal(address, destination);
/*    */   }
/*    */ 
/*    */   
/*    */   private void downloadInternal(URI address, File destination) throws Exception {
/* 47 */     OutputStream out = null;
/*    */     
/* 49 */     InputStream in = null;
/*    */     try {
/* 51 */       URL url = address.toURL();
/* 52 */       out = new BufferedOutputStream(new FileOutputStream(destination));
/* 53 */       URLConnection conn = url.openConnection();
/* 54 */       String userAgentValue = calculateUserAgent();
/* 55 */       conn.setRequestProperty("User-Agent", userAgentValue);
/* 56 */       in = conn.getInputStream();
/* 57 */       byte[] buffer = new byte[10000];
/*    */       
/* 59 */       long progressCounter = 0L; int numRead;
/* 60 */       while ((numRead = in.read(buffer)) != -1) {
/* 61 */         progressCounter += numRead;
/* 62 */         if (progressCounter / 20000L > 0L) {
/* 63 */           System.out.print(".");
/* 64 */           progressCounter -= 20000L;
/*    */         } 
/* 66 */         out.write(buffer, 0, numRead);
/*    */       } 
/*    */     } finally {
/* 69 */       System.out.println("");
/* 70 */       if (in != null) {
/* 71 */         in.close();
/*    */       }
/* 73 */       if (out != null) {
/* 74 */         out.close();
/*    */       }
/*    */     } 
/*    */   }
/*    */   
/*    */   private String calculateUserAgent() {
/* 80 */     String appVersion = this.applicationVersion;
/*    */     
/* 82 */     String javaVendor = System.getProperty("java.vendor");
/* 83 */     String javaVersion = System.getProperty("java.version");
/* 84 */     String javaVendorVersion = System.getProperty("java.vm.version");
/* 85 */     String osName = System.getProperty("os.name");
/* 86 */     String osVersion = System.getProperty("os.version");
/* 87 */     String osArch = System.getProperty("os.arch");
/* 88 */     return String.format("%s/%s (%s;%s;%s) (%s;%s;%s)", new Object[] { this.applicationName, appVersion, osName, osVersion, osArch, javaVendor, javaVersion, javaVendorVersion });
/*    */   }
/*    */   
/*    */   private static class SystemPropertiesProxyAuthenticator extends Authenticator {
/*    */     private SystemPropertiesProxyAuthenticator() {}
/*    */     
/*    */     protected PasswordAuthentication getPasswordAuthentication() {
/* 95 */       return new PasswordAuthentication(System.getProperty("http.proxyUser"), System.getProperty("http.proxyPassword", "").toCharArray());
/*    */     }
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\Download.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */