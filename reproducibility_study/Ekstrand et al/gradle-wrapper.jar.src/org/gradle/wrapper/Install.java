/*     */ package org.gradle.wrapper;
/*     */ 
/*     */ import java.io.BufferedOutputStream;
/*     */ import java.io.BufferedReader;
/*     */ import java.io.File;
/*     */ import java.io.FileOutputStream;
/*     */ import java.io.IOException;
/*     */ import java.io.InputStream;
/*     */ import java.io.InputStreamReader;
/*     */ import java.io.OutputStream;
/*     */ import java.net.URI;
/*     */ import java.util.ArrayList;
/*     */ import java.util.Enumeration;
/*     */ import java.util.Formatter;
/*     */ import java.util.List;
/*     */ import java.util.Locale;
/*     */ import java.util.concurrent.Callable;
/*     */ import java.util.zip.ZipEntry;
/*     */ import java.util.zip.ZipFile;
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ public class Install
/*     */ {
/*     */   public static final String DEFAULT_DISTRIBUTION_PATH = "wrapper/dists";
/*     */   private final IDownload download;
/*     */   private final PathAssembler pathAssembler;
/*  30 */   private final ExclusiveFileAccessManager exclusiveFileAccessManager = new ExclusiveFileAccessManager(120000, 200);
/*     */   
/*     */   public Install(IDownload download, PathAssembler pathAssembler) {
/*  33 */     this.download = download;
/*  34 */     this.pathAssembler = pathAssembler;
/*     */   }
/*     */   
/*     */   public File createDist(WrapperConfiguration configuration) throws Exception {
/*  38 */     final URI distributionUrl = configuration.getDistribution();
/*     */     
/*  40 */     PathAssembler.LocalDistribution localDistribution = this.pathAssembler.getDistribution(configuration);
/*  41 */     final File distDir = localDistribution.getDistributionDir();
/*  42 */     final File localZipFile = localDistribution.getZipFile();
/*     */     
/*  44 */     return this.exclusiveFileAccessManager.<File>access(localZipFile, new Callable<File>() {
/*     */           public File call() throws Exception {
/*  46 */             File markerFile = new File(localZipFile.getParentFile(), localZipFile.getName() + ".ok");
/*  47 */             if (distDir.isDirectory() && markerFile.isFile()) {
/*  48 */               return Install.this.getDistributionRoot(distDir, distDir.getAbsolutePath());
/*     */             }
/*     */             
/*  51 */             boolean needsDownload = !localZipFile.isFile();
/*     */             
/*  53 */             if (needsDownload) {
/*  54 */               File tmpZipFile = new File(localZipFile.getParentFile(), localZipFile.getName() + ".part");
/*  55 */               tmpZipFile.delete();
/*  56 */               System.out.println("Downloading " + distributionUrl);
/*  57 */               Install.this.download.download(distributionUrl, tmpZipFile);
/*  58 */               tmpZipFile.renameTo(localZipFile);
/*     */             } 
/*     */             
/*  61 */             List<File> topLevelDirs = Install.this.listDirs(distDir);
/*  62 */             for (File dir : topLevelDirs) {
/*  63 */               System.out.println("Deleting directory " + dir.getAbsolutePath());
/*  64 */               Install.this.deleteDir(dir);
/*     */             } 
/*  66 */             System.out.println("Unzipping " + localZipFile.getAbsolutePath() + " to " + distDir.getAbsolutePath());
/*  67 */             Install.this.unzip(localZipFile, distDir);
/*     */             
/*  69 */             File root = Install.this.getDistributionRoot(distDir, distributionUrl.toString());
/*  70 */             Install.this.setExecutablePermissions(root);
/*  71 */             markerFile.createNewFile();
/*     */             
/*  73 */             return root;
/*     */           }
/*     */         });
/*     */   }
/*     */   
/*     */   private File getDistributionRoot(File distDir, String distributionDescription) {
/*  79 */     List<File> dirs = listDirs(distDir);
/*  80 */     if (dirs.isEmpty()) {
/*  81 */       throw new RuntimeException(String.format("Gradle distribution '%s' does not contain any directories. Expected to find exactly 1 directory.", new Object[] { distributionDescription }));
/*     */     }
/*  83 */     if (dirs.size() != 1) {
/*  84 */       throw new RuntimeException(String.format("Gradle distribution '%s' contains too many directories. Expected to find exactly 1 directory.", new Object[] { distributionDescription }));
/*     */     }
/*  86 */     return dirs.get(0);
/*     */   }
/*     */   
/*     */   private List<File> listDirs(File distDir) {
/*  90 */     List<File> dirs = new ArrayList<File>();
/*  91 */     if (distDir.exists()) {
/*  92 */       for (File file : distDir.listFiles()) {
/*  93 */         if (file.isDirectory()) {
/*  94 */           dirs.add(file);
/*     */         }
/*     */       } 
/*     */     }
/*  98 */     return dirs;
/*     */   }
/*     */   
/*     */   private void setExecutablePermissions(File gradleHome) {
/* 102 */     if (isWindows()) {
/*     */       return;
/*     */     }
/* 105 */     File gradleCommand = new File(gradleHome, "bin/gradle");
/* 106 */     String errorMessage = null;
/*     */     try {
/* 108 */       ProcessBuilder pb = new ProcessBuilder(new String[] { "chmod", "755", gradleCommand.getCanonicalPath() });
/* 109 */       Process p = pb.start();
/* 110 */       if (p.waitFor() == 0) {
/* 111 */         System.out.println("Set executable permissions for: " + gradleCommand.getAbsolutePath());
/*     */       } else {
/* 113 */         BufferedReader is = new BufferedReader(new InputStreamReader(p.getInputStream()));
/* 114 */         Formatter stdout = new Formatter();
/*     */         String line;
/* 116 */         while ((line = is.readLine()) != null) {
/* 117 */           stdout.format("%s%n", new Object[] { line });
/*     */         } 
/* 119 */         errorMessage = stdout.toString();
/*     */       } 
/* 121 */     } catch (IOException e) {
/* 122 */       errorMessage = e.getMessage();
/* 123 */     } catch (InterruptedException e) {
/* 124 */       errorMessage = e.getMessage();
/*     */     } 
/* 126 */     if (errorMessage != null) {
/* 127 */       System.out.println("Could not set executable permissions for: " + gradleCommand.getAbsolutePath());
/* 128 */       System.out.println("Please do this manually if you want to use the Gradle UI.");
/*     */     } 
/*     */   }
/*     */   
/*     */   private boolean isWindows() {
/* 133 */     String osName = System.getProperty("os.name").toLowerCase(Locale.US);
/* 134 */     if (osName.indexOf("windows") > -1) {
/* 135 */       return true;
/*     */     }
/* 137 */     return false;
/*     */   }
/*     */   
/*     */   private boolean deleteDir(File dir) {
/* 141 */     if (dir.isDirectory()) {
/* 142 */       String[] children = dir.list();
/* 143 */       for (int i = 0; i < children.length; i++) {
/* 144 */         boolean success = deleteDir(new File(dir, children[i]));
/* 145 */         if (!success) {
/* 146 */           return false;
/*     */         }
/*     */       } 
/*     */     } 
/*     */ 
/*     */     
/* 152 */     return dir.delete();
/*     */   }
/*     */ 
/*     */   
/*     */   private void unzip(File zip, File dest) throws IOException {
/* 157 */     ZipFile zipFile = new ZipFile(zip);
/*     */     
/*     */     try {
/* 160 */       Enumeration<? extends ZipEntry> entries = zipFile.entries();
/*     */       
/* 162 */       while (entries.hasMoreElements()) {
/* 163 */         ZipEntry entry = entries.nextElement();
/*     */         
/* 165 */         if (entry.isDirectory()) {
/* 166 */           (new File(dest, entry.getName())).mkdirs();
/*     */           
/*     */           continue;
/*     */         } 
/* 170 */         OutputStream outputStream = new BufferedOutputStream(new FileOutputStream(new File(dest, entry.getName())));
/*     */         try {
/* 172 */           copyInputStream(zipFile.getInputStream(entry), outputStream);
/*     */         } finally {
/* 174 */           outputStream.close();
/*     */         } 
/*     */       } 
/*     */     } finally {
/* 178 */       zipFile.close();
/*     */     } 
/*     */   }
/*     */   
/*     */   private void copyInputStream(InputStream in, OutputStream out) throws IOException {
/* 183 */     byte[] buffer = new byte[1024];
/*     */     
/*     */     int len;
/* 186 */     while ((len = in.read(buffer)) >= 0) {
/* 187 */       out.write(buffer, 0, len);
/*     */     }
/*     */     
/* 190 */     in.close();
/* 191 */     out.close();
/*     */   }
/*     */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\Install.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */