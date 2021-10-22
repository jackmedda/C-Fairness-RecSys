/*    */ package org.gradle.wrapper;
/*    */ 
/*    */ import java.io.File;
/*    */ import java.io.FileInputStream;
/*    */ import java.io.IOException;
/*    */ import java.util.HashMap;
/*    */ import java.util.Map;
/*    */ import java.util.Properties;
/*    */ import java.util.regex.Matcher;
/*    */ import java.util.regex.Pattern;
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
/*    */ public class SystemPropertiesHandler
/*    */ {
/*    */   public static Map<String, String> getSystemProperties(File propertiesFile) {
/* 30 */     Map<String, String> propertyMap = new HashMap<String, String>();
/* 31 */     if (!propertiesFile.isFile()) {
/* 32 */       return propertyMap;
/*    */     }
/* 34 */     Properties properties = new Properties();
/*    */     try {
/* 36 */       FileInputStream inStream = new FileInputStream(propertiesFile);
/*    */       try {
/* 38 */         properties.load(inStream);
/*    */       } finally {
/* 40 */         inStream.close();
/*    */       } 
/* 42 */     } catch (IOException e) {
/* 43 */       throw new RuntimeException("Error when loading properties file=" + propertiesFile, e);
/*    */     } 
/*    */     
/* 46 */     Pattern pattern = Pattern.compile("systemProp\\.(.*)");
/* 47 */     for (Object argument : properties.keySet()) {
/* 48 */       Matcher matcher = pattern.matcher(argument.toString());
/* 49 */       if (matcher.find()) {
/* 50 */         String key = matcher.group(1);
/* 51 */         if (key.length() > 0) {
/* 52 */           propertyMap.put(key, properties.get(argument).toString());
/*    */         }
/*    */       } 
/*    */     } 
/* 56 */     return propertyMap;
/*    */   }
/*    */ }


/* Location:              C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\All the cool kids\gradle\wrapper\gradle-wrapper.jar!\org\gradle\wrapper\SystemPropertiesHandler.class
 * Java compiler version: 5 (49.0)
 * JD-Core Version:       1.1.3
 */