/*
 * PHASE 6 -- STM32 firmware: MPU6050 -> USART2 CSV streaming.
 * =======================================================================
 * VERIFICATION STATUS: written and statically reviewed only. NOT
 * compiled or run on real hardware in this environment -- no physical
 * STM32 board is available here. Build and test this yourself before
 * trusting it (see the Phase 6 write-up's testing section).
 *
 * This file is NOT a standalone buildable project by itself. It is a
 * copy-paste reference for the specific "USER CODE" sections inside the
 * main.c that STM32CubeIDE generates for you, AFTER you have:
 *   1. Created a new STM32CubeIDE project for your board.
 *   2. Enabled and configured I2C1 (for the MPU6050) and USART2 (for
 *      serial output to the PC) in the .ioc pinout/configuration tool.
 *   3. Added mpu6050.h / mpu6050.c to your project's Core/Inc and
 *      Core/Src folders.
 * See the Phase 6 write-up for the exact CubeIDE configuration steps
 * and wiring table.
 *
 * CubeIDE preserves anything you put between a "USER CODE BEGIN X" /
 * "USER CODE END X" pair across re-generation, so paste each block
 * below into the matching pair in your generated main.c.
 */

/* USER CODE BEGIN Includes */
#include "mpu6050.h"
#include <stdio.h>
#include <string.h>
/* USER CODE END Includes */


/* USER CODE BEGIN PV */
static MPU6050_Reading g_reading;
static char g_tx_buf[96];
/* USER CODE END PV */


/* USER CODE BEGIN 2 */
/* Place this AFTER the generated calls to MX_I2C1_Init() and
 * MX_USART2_UART_Init() in main(). */
if (MPU6050_Init(&hi2c1) != HAL_OK)
{
    /* WHO_AM_I check or a wake-up/config write failed -- almost always
     * a wiring problem (SDA/SCL swapped, no pull-ups, wrong I2C
     * address) rather than a firmware bug. Halting here (via
     * Error_Handler(), CubeIDE's generated default trap) makes the
     * failure visible instead of silently streaming zeros/garbage. */
    Error_Handler();
}
/* USER CODE END 2 */


/* USER CODE BEGIN WHILE */
/* Place this INSIDE the generated `while (1) { ... }` main loop. */
if (MPU6050_ReadScaled(&hi2c1, &g_reading) == HAL_OK)
{
    /* Serial line format (see the Phase 6 write-up for the full spec):
     *   timestamp_ms,accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps
     * Example: 1234,0.02,-0.01,1.01,2.3,-1.7,4.2
     * timestamp_ms is HAL_GetTick() (milliseconds since boot) -- useful
     * for firmware-side debugging; the Python side does not use it as
     * an ML feature. */
    int len = snprintf(g_tx_buf, sizeof(g_tx_buf),
        "%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\r\n",
        (unsigned long)HAL_GetTick(),
        g_reading.accel_x_g, g_reading.accel_y_g, g_reading.accel_z_g,
        g_reading.gyro_x_dps, g_reading.gyro_y_dps, g_reading.gyro_z_dps);

    if (len > 0)
    {
        HAL_UART_Transmit(&huart2, (uint8_t *)g_tx_buf, (uint16_t)len, HAL_MAX_DELAY);
    }
}
/* else: this read failed (I2C hiccup) -- skip silently and try again
 * next loop iteration rather than sending a malformed/partial line;
 * the Python-side parser treats an occasional missing line as normal. */

HAL_Delay(100);  /* ~10 Hz, matching the Python pipeline's expected
                  * 10-sample-per-second / 1-second-window design. */
/* USER CODE END WHILE */
