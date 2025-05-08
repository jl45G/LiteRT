// In the AI pack's build.gradle file:
plugins { id("com.android.ai-pack") }

aiPack {
  packName = "selfie_multiclass_ai_pack"
  dynamicDelivery { deliveryType = "on-demand" }
}
