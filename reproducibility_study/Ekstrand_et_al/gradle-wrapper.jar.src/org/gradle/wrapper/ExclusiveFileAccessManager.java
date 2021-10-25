/*    */ package org.gradle.wrapper;
/*    */ 
/*    */ import java.io.Closeable;
/*    */ import java.io.File;
/*    */ import java.io.RandomAccessFile;
/*    */ import java.nio.channels.FileChannel;
/*    */ import java.nio.channels.FileLock;
/*    */ import java.util.concurrent.Callable;
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
/*    */ 
/*    */ 
/*    */ public class ExclusiveFileAccessManager
/*    */ {
/*    */   public static final String LOCK_FILE_SUFFIX = ".lck";
/*    */   private final int timeoutMs;
/*    */   private final int pollIntervalMs;
/*    */   
/*    */   public ExclusiveFileAccessManager(int timeoutMs, int pollIntervalMs) {
/* 34 */     this.timeoutMs = timeoutMs;
/* 35 */     this.pollIntervalMs = pollIntervalMs;
/*    */   }
/*    */   
/*    */   public <T> T access(File exclusiveFile, Callable<T> task) throws Exception {
/* 39 */     File lockFile = new File(exclusiveFile.getParentFile(), exclusiveFile.getName() + ".lck");
/* 40 */     lockFile.getParentFile().mkdirs();
/* 41 */     RandomAccessFile randomAccessFile = null;
/* 42 */     FileChannel channel = null;
/*    */     
/*    */     try {
/* 45 */       long startAt = System.currentTimeMillis();
/* 46 */       FileLock lock = null;
/*    */       
/* 48 */       while (lock == null && System.currentTimeMillis() < startAt + this.timeoutMs) {
/* 49 */         randomAccessFile = new RandomAccessFile(lockFile, "rw");
/* 50 */         channel = randomAccessFile.getChannel();
/* 51 */         lock = channel.tryLock();
/*    */         
/* 53 */         if (lock == null) {
/* 54 */           maybeCloseQuietly(channel);
/* 55 */           maybeCloseQuietly(randomAccessFile);
/* 56 */           Thread.sleep(this.pollIntervalMs);
/*    */ 
/*    */ 
/*    */         
/*    */         }
/*    */ 
/*    */ 
/*    */ 
/*    */       
/*    */       }
/*    */ 
/*    */ 
/*    */ 
/*    */     
/*    */     }
/*    */     finally {
/*    */ 
/*    */ 
/*    */       
/* 75 */       maybeCloseQuietly(channel);
/* 76 */       maybeCloseQuietly(randomAccessFile);
/*    */     } 
/*    */   }
/*    */   
/*    */   private static void maybeCloseQuietly(Closeable closeable) {
/* 81 */     if (closeable != null)
/*    */       try {
/* 83 */         closeable.close();
/* 84 */       } catch (Exception ignore) {} 
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\ExclusiveFileAccessManager.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */